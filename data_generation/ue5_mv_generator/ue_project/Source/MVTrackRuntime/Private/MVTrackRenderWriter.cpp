// MVTrackRenderWriter.cpp — RGB/Depth capture with Xvfb display
#include "MVTrackRenderWriter.h"
#include "Components/SceneCaptureComponent2D.h"
#include "Engine/TextureRenderTarget2D.h"
#include "Misc/FileHelper.h"
#include "HAL/PlatformFileManager.h"
#include "ImageUtils.h"
#include "Rendering/Texture2DResource.h"
#include "RHICommandList.h"
#include "Components/StaticMeshComponent.h"
#include "Materials/Material.h"
#include "Materials/MaterialInstanceDynamic.h"
#include "EngineUtils.h"

bool UMVTrackRenderWriter::InitRenderTargets(int32 ResX, int32 ResY)
{
    Width = ResX;
    Height = ResY;

    RGBTarget = NewObject<UTextureRenderTarget2D>(this);
    RGBTarget->InitCustomFormat(ResX, ResY, PF_B8G8R8A8, false);
    RGBTarget->TargetGamma = 2.2f;
    RGBTarget->UpdateResourceImmediate(true);

    DepthTarget = NewObject<UTextureRenderTarget2D>(this);
    DepthTarget->InitCustomFormat(ResX, ResY, PF_FloatRGBA, false);
    DepthTarget->TargetGamma = 1.0f;
    DepthTarget->UpdateResourceImmediate(true);

    MaskTarget = NewObject<UTextureRenderTarget2D>(this);
    MaskTarget->InitCustomFormat(ResX, ResY, PF_B8G8R8A8, false);
    MaskTarget->TargetGamma = 1.0f;
    MaskTarget->ClearColor = FLinearColor::Black;
    MaskTarget->UpdateResourceImmediate(true);

    UE_LOG(LogTemp, Log, TEXT("[MVTrack] Render targets: %dx%d"), ResX, ResY);
    return RGBTarget && DepthTarget && MaskTarget;
}

bool UMVTrackRenderWriter::CaptureRGB(USceneCaptureComponent2D* CaptureComp, const FString& OutputPath)
{
    if (!CaptureComp || !RGBTarget) return false;

    CaptureComp->TextureTarget = RGBTarget;
    CaptureComp->CaptureSource = ESceneCaptureSource::SCS_FinalColorLDR;
    CaptureComp->CaptureScene();

    // Read pixels on game thread (works with Xvfb display)
    TArray<FColor> Bitmap;
    FReadSurfaceDataFlags ReadFlags;
    ReadFlags.SetLinearToGamma(true);

    FTextureRenderTargetResource* RTResource = RGBTarget->GameThread_GetRenderTargetResource();
    if (!RTResource)
    {
        UE_LOG(LogTemp, Warning, TEXT("[MVTrack] No RT resource"));
        return false;
    }

    RTResource->ReadPixels(Bitmap, ReadFlags);

    if (Bitmap.Num() == Width * Height)
    {
        TArray<uint8> PNGData;
        FImageUtils::CompressImageArray(Width, Height, Bitmap, PNGData);
        IFileManager::Get().MakeDirectory(*FPaths::GetPath(OutputPath), true);
        bool bSaved = FFileHelper::SaveArrayToFile(PNGData, *OutputPath);
        UE_LOG(LogTemp, Log, TEXT("[MVTrack] RGB saved: %s (%d bytes)"), *OutputPath, PNGData.Num());
        return bSaved;
    }

    UE_LOG(LogTemp, Warning, TEXT("[MVTrack] ReadPixels: got %d, expected %d"), Bitmap.Num(), Width * Height);
    return false;
}

bool UMVTrackRenderWriter::CaptureDepth(USceneCaptureComponent2D* CaptureComp, const FString& OutputPath)
{
    if (!CaptureComp || !DepthTarget) return false;

    CaptureComp->TextureTarget = DepthTarget;
    CaptureComp->CaptureSource = ESceneCaptureSource::SCS_SceneDepth;
    CaptureComp->CaptureScene();

    // Restore RGB
    CaptureComp->TextureTarget = RGBTarget;
    CaptureComp->CaptureSource = ESceneCaptureSource::SCS_FinalColorLDR;

    TArray<FColor> Bitmap;
    FReadSurfaceDataFlags ReadFlags;
    FTextureRenderTargetResource* RTResource = DepthTarget->GameThread_GetRenderTargetResource();
    if (!RTResource) return false;

    RTResource->ReadPixels(Bitmap, ReadFlags);

    if (Bitmap.Num() == Width * Height)
    {
        // Encode depth as 16-bit PNG
        TArray<FColor> DepthColors;
        DepthColors.SetNum(Width * Height);
        for (int32 i = 0; i < Bitmap.Num(); i++)
        {
            float Depth = Bitmap[i].R / 255.0f * 100.0f;
            uint16 D16 = FMath::Clamp((uint16)(Depth * 100.0f), (uint16)0, (uint16)65535);
            DepthColors[i] = FColor(D16 >> 8, D16 & 0xFF, 0, 255);
        }
        TArray<uint8> PNGData;
        FImageUtils::CompressImageArray(Width, Height, DepthColors, PNGData);
        FString PNGPath = OutputPath.Replace(TEXT(".exr"), TEXT(".png"));
        IFileManager::Get().MakeDirectory(*FPaths::GetPath(PNGPath), true);
        return FFileHelper::SaveArrayToFile(PNGData, *PNGPath);
    }

    return false;
}

FMVTrackMaskStats UMVTrackRenderWriter::CaptureMask(
    USceneCaptureComponent2D* CaptureComp,
    AActor* TargetActor,
    const FString& OutputPath)
{
    FMVTrackMaskStats Stats;
    if (!CaptureComp || !MaskTarget || !TargetActor)
    {
        return Stats;
    }

    UMaterial* Base = LoadObject<UMaterial>(
        UMaterial::StaticClass(),
        TEXT("/Engine/BasicShapes/BasicShapeMaterial"));
    if (!Base)
    {
        return Stats;
    }

    struct FRestoreMaterial
    {
        UStaticMeshComponent* Comp = nullptr;
        TArray<UMaterialInterface*> Materials;
    };

    TArray<FRestoreMaterial> Restore;
    Restore.Reserve(128);

    UMaterialInstanceDynamic* Black = UMaterialInstanceDynamic::Create(Base, this);
    UMaterialInstanceDynamic* White = UMaterialInstanceDynamic::Create(Base, this);
    if (!Black || !White)
    {
        return Stats;
    }
    Black->SetVectorParameterValue(TEXT("Color"), FLinearColor::Black);
    Black->SetVectorParameterValue(TEXT("BaseColor"), FLinearColor::Black);
    Black->SetScalarParameterValue(TEXT("Roughness"), 1.0f);
    White->SetVectorParameterValue(TEXT("Color"), FLinearColor::White);
    White->SetVectorParameterValue(TEXT("BaseColor"), FLinearColor::White);
    White->SetScalarParameterValue(TEXT("Roughness"), 1.0f);

    for (TActorIterator<AActor> It(GetWorld()); It; ++It)
    {
        AActor* Actor = *It;
        if (!Actor) continue;
        TArray<UStaticMeshComponent*> Comps;
        Actor->GetComponents<UStaticMeshComponent>(Comps);
        for (UStaticMeshComponent* Comp : Comps)
        {
            if (!Comp) continue;
            FRestoreMaterial R;
            R.Comp = Comp;
            const int32 NumSlots = FMath::Max(1, Comp->GetNumMaterials());
            for (int32 Slot = 0; Slot < NumSlots; ++Slot)
            {
                R.Materials.Add(Comp->GetMaterial(Slot));
                Comp->SetMaterial(Slot, Actor == TargetActor ? White : Black);
            }
            Restore.Add(MoveTemp(R));
        }
    }

    CaptureComp->TextureTarget = MaskTarget;
    CaptureComp->CaptureSource = ESceneCaptureSource::SCS_BaseColor;
    CaptureComp->CaptureScene();

    TArray<FColor> Bitmap;
    FReadSurfaceDataFlags ReadFlags;
    ReadFlags.SetLinearToGamma(false);
    FTextureRenderTargetResource* RTResource = MaskTarget->GameThread_GetRenderTargetResource();
    if (RTResource)
    {
        RTResource->ReadPixels(Bitmap, ReadFlags);
    }

    for (FRestoreMaterial& R : Restore)
    {
        if (!R.Comp) continue;
        for (int32 Slot = 0; Slot < R.Materials.Num(); ++Slot)
        {
            R.Comp->SetMaterial(Slot, R.Materials[Slot]);
        }
    }

    CaptureComp->TextureTarget = RGBTarget;
    CaptureComp->CaptureSource = ESceneCaptureSource::SCS_FinalColorLDR;

    if (Bitmap.Num() != Width * Height)
    {
        return Stats;
    }

    int32 MinX = Width;
    int32 MinY = Height;
    int32 MaxX = -1;
    int32 MaxY = -1;
    TArray<FColor> MaskColors;
    MaskColors.SetNum(Bitmap.Num());
    for (int32 Y = 0; Y < Height; ++Y)
    {
        for (int32 X = 0; X < Width; ++X)
        {
            const int32 Index = Y * Width + X;
            const FColor& P = Bitmap[Index];
            const bool bVisible = (P.R > 127 && P.G > 127 && P.B > 127);
            MaskColors[Index] = bVisible ? FColor::White : FColor::Black;
            if (bVisible)
            {
                ++Stats.VisiblePixels;
                MinX = FMath::Min(MinX, X);
                MinY = FMath::Min(MinY, Y);
                MaxX = FMath::Max(MaxX, X);
                MaxY = FMath::Max(MaxY, Y);
            }
        }
    }

    if (Stats.VisiblePixels > 0)
    {
        Stats.bValid = true;
        Stats.VisibleBBox = FVector4f(
            float(MinX),
            float(MinY),
            float(MaxX - MinX + 1),
            float(MaxY - MinY + 1));
    }

    TArray<uint8> PNGData;
    FImageUtils::CompressImageArray(Width, Height, MaskColors, PNGData);
    IFileManager::Get().MakeDirectory(*FPaths::GetPath(OutputPath), true);
    FFileHelper::SaveArrayToFile(PNGData, *OutputPath);
    return Stats;
}

bool UMVTrackRenderWriter::CaptureAll(
    USceneCaptureComponent2D* CaptureComp,
    AActor* TargetActor,
    const FString& FrameDir,
    int32 FrameIndex,
    FMVTrackMaskStats* OutMaskStats)
{
    if (!CaptureComp) return false;

    FString FrameStr = FString::Printf(TEXT("%06d"), FrameIndex);

    FString RGBDir = FrameDir / TEXT("rgb");
    FString DepthDir = FrameDir / TEXT("depth");
    FString MaskDir = FrameDir / TEXT("mask");
    IFileManager::Get().MakeDirectory(*RGBDir, true);
    IFileManager::Get().MakeDirectory(*DepthDir, true);
    IFileManager::Get().MakeDirectory(*MaskDir, true);

    // Capture RGB
    FString RGBPath = RGBDir / FrameStr + TEXT(".png");
    bool bRGB = CaptureRGB(CaptureComp, RGBPath);

    // Capture Depth
    FString DepthPath = DepthDir / FrameStr + TEXT(".exr");
    bool bDepth = CaptureDepth(CaptureComp, DepthPath);

    FString MaskPath = MaskDir / FrameStr + TEXT(".png");
    FMVTrackMaskStats MaskStats = CaptureMask(CaptureComp, TargetActor, MaskPath);
    if (OutMaskStats)
    {
        *OutMaskStats = MaskStats;
    }

    return bRGB;
}

void UMVTrackRenderWriter::Cleanup()
{
    if (RGBTarget) { RGBTarget->ReleaseResource(); RGBTarget = nullptr; }
    if (DepthTarget) { DepthTarget->ReleaseResource(); DepthTarget = nullptr; }
    if (MaskTarget) { MaskTarget->ReleaseResource(); MaskTarget = nullptr; }
}
