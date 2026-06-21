#include "MVTrackCameraManager.h"
#include "Components/SceneCaptureComponent2D.h"
#include "Components/SceneComponent.h"
#include "Engine/StaticMeshActor.h"
#include "Camera/CameraComponent.h"
#include "Kismet/KismetMathLibrary.h"
#include "Engine/World.h"
#include "Serialization/JsonWriter.h"
#include "Serialization/JsonSerializer.h"
#include "HAL/PlatformFileManager.h"
#include "Misc/FileHelper.h"

void UMVTrackCameraManager::SetupCameras(
    const FMVTrackJobManifest& Manifest,
    const FVector& TrajectoryCenter,
    const FVector& TargetInitialPos,
    FRandomStream& RNG)
{
    NumCameras = Manifest.NumCameras;
    Cameras.Empty();

    float RadiusMin = 220.0f;
    float RadiusMax = 340.0f;
    float HeightMin = 75.0f;
    float HeightMax = 165.0f;

    for (int32 i = 0; i < NumCameras; i++)
    {
        FMVTrackCameraData CamData;
        CamData.CameraId = FString::Printf(TEXT("cam_%03d"), i);

        FVector CamPos = PlaceCameraOnHemisphere(
            i, NumCameras, TrajectoryCenter,
            RNG.FRandRange(RadiusMin, RadiusMax),
            HeightMin, HeightMax, 360.0f, -15.0f, 40.0f, RNG);

        // Compute look-at rotation before spawning
        FRotator LookAt = UKismetMathLibrary::FindLookAtRotation(CamPos, TargetInitialPos);

        FActorSpawnParameters SpawnParams;
        SpawnParams.SpawnCollisionHandlingOverride = ESpawnActorCollisionHandlingMethod::AlwaysSpawn;
        AActor* CamActor = GetWorld()->SpawnActor<AActor>(CamPos, LookAt, SpawnParams);

        // Add a root scene component so the actor has proper transform
        USceneComponent* RootComp = NewObject<USceneComponent>(CamActor);
        RootComp->SetWorldLocation(CamPos);
        RootComp->SetWorldRotation(LookAt);
        CamActor->SetRootComponent(RootComp);
        RootComp->RegisterComponent();

        USceneCaptureComponent2D* CaptureComp = NewObject<USceneCaptureComponent2D>(CamActor);
        CaptureComp->SetupAttachment(RootComp);
        CaptureComp->RegisterComponent();
        CaptureComp->CaptureSource = ESceneCaptureSource::SCS_FinalColorLDR;
        CaptureComp->bCaptureEveryFrame = false;
        CaptureComp->bCaptureOnMovement = false;

        float FocalLengthMM = RNG.FRandRange(26.0f, 34.0f);
        float FOV = FMath::RadiansToDegrees(2.0f * FMath::Atan(36.0f / (2.0f * FocalLengthMM)));
        CaptureComp->FOVAngle = FOV;

        CamData.CameraActor = CamActor;
        CamData.CaptureComp = CaptureComp;
        CamData.Calibration.CameraId = CamData.CameraId;
        CamData.Calibration.ResolutionX = Manifest.ResolutionX;
        CamData.Calibration.ResolutionY = Manifest.ResolutionY;
        CamData.Calibration.FocalLengthMM = FocalLengthMM;
        CamData.Calibration.SensorWidthMM = 36.0f;
        CamData.Calibration.SensorHeightMM = 20.25f;
        ComputeCalibration(CamData, Manifest.ResolutionX, Manifest.ResolutionY);
        Cameras.Add(MoveTemp(CamData));

        UE_LOG(LogTemp, Log, TEXT("[MVTrack] Camera %s at %s looking at %s"),
            *CamData.CameraId, *CamPos.ToString(), *TargetInitialPos.ToString());
    }
    UE_LOG(LogTemp, Log, TEXT("[MVTrack] Spawned %d cameras"), NumCameras);
}

FVector UMVTrackCameraManager::PlaceCameraOnHemisphere(
    int32 Index, int32 Total, const FVector& Center, float Radius,
    float HeightMin, float HeightMax, float YawCoverageDeg,
    float PitchMinDeg, float PitchMaxDeg, FRandomStream& RNG)
{
    float AzimuthStep = YawCoverageDeg / FMath::Max(Total, 1);
    float Azimuth = Index * AzimuthStep + RNG.FRandRange(-AzimuthStep * 0.3f, AzimuthStep * 0.3f);
    float Elevation = RNG.FRandRange(PitchMinDeg, PitchMaxDeg);
    float AzRad = FMath::DegreesToRadians(Azimuth);
    float ElRad = FMath::DegreesToRadians(Elevation);
    float X = Center.X + Radius * FMath::Cos(ElRad) * FMath::Cos(AzRad);
    float Y = Center.Y + Radius * FMath::Cos(ElRad) * FMath::Sin(AzRad);
    float Z = FMath::Lerp(HeightMin, HeightMax, (FMath::Sin(ElRad) + 1.0f) * 0.5f);
    X = FMath::Clamp(X, -460.0f, 460.0f);
    Y = FMath::Clamp(Y, -460.0f, 460.0f);
    Z = FMath::Clamp(Z, 80.0f, 240.0f);
    return FVector(X, Y, Z);
}

void UMVTrackCameraManager::ComputeCalibration(FMVTrackCameraData& CamData, int32 ResX, int32 ResY)
{
    float FL = CamData.Calibration.FocalLengthMM;
    float Sw = CamData.Calibration.SensorWidthMM;
    float Sh = CamData.Calibration.SensorHeightMM;
    float fx = (FL / Sw) * ResX;
    float fy = (FL / Sh) * ResY;
    float cx = ResX * 0.5f;
    float cy = ResY * 0.5f;
    CamData.Calibration.PrincipalPoint = FVector2D(cx, cy);
    CamData.Calibration.KMatrix = {fx, 0.0f, cx, 0.0f, fy, cy, 0.0f, 0.0f, 1.0f};
    if (CamData.CameraActor)
    {
        CamData.Calibration.CameraToWorld = CamData.CameraActor->GetActorTransform();
        CamData.Calibration.WorldToCamera = CamData.CameraActor->GetActorTransform().Inverse();
    }
}

void UMVTrackCameraManager::CaptureAllCameras(int32 FrameIndex, const FString& OutputDir)
{
    for (auto& Cam : Cameras)
    {
        if (!Cam.CaptureComp) continue;
        FString CamDir = FString::Printf(TEXT("%s/frames/%s"), *OutputDir, *Cam.CameraId);
        IFileManager::Get().MakeDirectory(*CamDir, true);
        Cam.CaptureComp->CaptureScene();
    }
}

TArray<FMVTrackCameraCalibration> UMVTrackCameraManager::GetAllCalibrations() const
{
    TArray<FMVTrackCameraCalibration> Result;
    for (const auto& Cam : Cameras) { Result.Add(Cam.Calibration); }
    return Result;
}

void UMVTrackCameraManager::SaveCalibrations(const FString& OutputDir) const
{
    for (const auto& Cam : Cameras)
    {
        TSharedPtr<FJsonObject> Json = MakeShareable(new FJsonObject);
        Json->SetStringField(TEXT("camera_id"), Cam.CameraId);
        Json->SetNumberField(TEXT("resolution_x"), Cam.Calibration.ResolutionX);
        Json->SetNumberField(TEXT("resolution_y"), Cam.Calibration.ResolutionY);
        Json->SetNumberField(TEXT("focal_length_mm"), Cam.Calibration.FocalLengthMM);

        TArray<TSharedPtr<FJsonValue>> KArr;
        for (float v : Cam.Calibration.KMatrix)
            KArr.Add(MakeShareable(new FJsonValueNumber(v)));
        Json->SetArrayField(TEXT("K_matrix_row_major"), KArr);

        FTransform C2W = Cam.Calibration.CameraToWorld;
        TSharedPtr<FJsonObject> C2WJson = MakeShareable(new FJsonObject);
        C2WJson->SetNumberField(TEXT("tx"), C2W.GetLocation().X);
        C2WJson->SetNumberField(TEXT("ty"), C2W.GetLocation().Y);
        C2WJson->SetNumberField(TEXT("tz"), C2W.GetLocation().Z);
        FQuat Q = C2W.GetRotation();
        C2WJson->SetNumberField(TEXT("qx"), Q.X);
        C2WJson->SetNumberField(TEXT("qy"), Q.Y);
        C2WJson->SetNumberField(TEXT("qz"), Q.Z);
        C2WJson->SetNumberField(TEXT("qw"), Q.W);
        Json->SetObjectField(TEXT("camera_to_world"), C2WJson);

        FString JsonStr;
        TSharedRef<TJsonWriter<>> Writer = TJsonWriterFactory<>::Create(&JsonStr);
        FJsonSerializer::Serialize(Json.ToSharedRef(), Writer);

        FString FilePath = FString::Printf(TEXT("%s/cameras/%s.json"), *OutputDir, *Cam.CameraId);
        IFileManager::Get().MakeDirectory(*FPaths::GetPath(FilePath), true);
        FFileHelper::SaveStringToFile(JsonStr, *FilePath);
    }
}

int32 UMVTrackCameraManager::CountVisibleCameras(const FVector& TargetLocation, float TargetRadius) const
{
    int32 Count = 0;
    for (const auto& Cam : Cameras)
    {
        if (!Cam.CameraActor) continue;
        FVector Start = Cam.CameraActor->GetActorLocation();
        FHitResult Hit;
        FCollisionQueryParams Params;
        bool bHit = GetWorld()->LineTraceSingleByChannel(Hit, Start, TargetLocation, ECC_Visibility, Params);
        if (!bHit || (Hit.Location - Start).Size() >= (TargetLocation - Start).Size() - TargetRadius)
            Count++;
    }
    return Count;
}

TArray<bool> UMVTrackCameraManager::CheckOcclusion(const FVector& TargetLocation, float TargetRadius) const
{
    TArray<bool> Occluded;
    for (const auto& Cam : Cameras)
    {
        bool bOccluded = false;
        if (Cam.CameraActor)
        {
            FVector Start = Cam.CameraActor->GetActorLocation();
            FHitResult Hit;
            FCollisionQueryParams Params;
            bool bHit = GetWorld()->LineTraceSingleByChannel(Hit, Start, TargetLocation, ECC_Visibility, Params);
            if (bHit && (Hit.Location - Start).Size() < (TargetLocation - Start).Size() - TargetRadius)
                bOccluded = true;
        }
        Occluded.Add(bOccluded);
    }
    return Occluded;
}

void UMVTrackCameraManager::UpdateCameraLookAt(const FVector& TargetLocation)
{
    for (auto& Cam : Cameras)
    {
        if (!Cam.CameraActor || !Cam.CaptureComp) continue;
        FVector CamPos = Cam.CameraActor->GetActorLocation();
        FRotator LookAt = UKismetMathLibrary::FindLookAtRotation(CamPos, TargetLocation);
        Cam.CameraActor->SetActorRotation(LookAt);
        Cam.CaptureComp->SetWorldRotation(LookAt);
        ComputeCalibration(Cam, Cam.Calibration.ResolutionX, Cam.Calibration.ResolutionY);
    }
}

USceneCaptureComponent2D* UMVTrackCameraManager::GetCaptureComp(int32 Index) const
{
    if (Index >= 0 && Index < Cameras.Num())
        return Cameras[Index].CaptureComp;
    return nullptr;
}
