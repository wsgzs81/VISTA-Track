#include "MVTrackSequenceRunner.h"
#include "MVTrackJobConfig.h"
#include "MVTrackMaterialLibrary.h"
#include "MVTrackCameraManager.h"
#include "MVTrackTargetController.h"
#include "MVTrackAnnotationWriter.h"
#include "MVTrackRenderWriter.h"
#include "Engine/World.h"
#include "Engine/StaticMesh.h"
#include "Components/StaticMeshComponent.h"
#include "Components/DirectionalLightComponent.h"
#include "Components/SkyLightComponent.h"
#include "Components/PointLightComponent.h"
#include "Components/ExponentialHeightFogComponent.h"
#include "Engine/StaticMeshActor.h"
#include "Engine/DirectionalLight.h"
#include "Engine/SkyLight.h"
#include "Engine/PointLight.h"
#include "Engine/ExponentialHeightFog.h"
#include "Components/SceneCaptureComponent2D.h"
#include "Kismet/GameplayStatics.h"
#include "Kismet/KismetMathLibrary.h"
#include "Misc/FileHelper.h"
#include "HAL/PlatformFileManager.h"
#include "EngineUtils.h"

static const TCHAR* FurniturePaths[] = {
    TEXT("/Game/Assets/Realistic/chinese_stool/model"),
    TEXT("/Game/Assets/Realistic/coffee_table_round_01/model"),
    TEXT("/Game/Assets/Realistic/painted_wooden_cabinet_02/model"),
    TEXT("/Game/Assets/Realistic/painted_wooden_table/model"),
    TEXT("/Game/Assets/Realistic/round_wooden_table_01/model"),
    TEXT("/Game/Assets/Realistic/small_wooden_table_01/model"),
    TEXT("/Game/Assets/Realistic/Television_01/model"),
    TEXT("/Game/Assets/Realistic/wicker_basket_01/model"),
    TEXT("/Game/Assets/Realistic/wooden_picnic_table/model"),
};
static const int32 NumFurniture = 9;

namespace
{
FVector GroundLocationNear(const FVector& Center, FRandomStream& RNG, float MinR, float MaxR)
{
    const float Angle = RNG.FRandRange(0.0f, 2.0f * PI);
    const float Radius = RNG.FRandRange(MinR, MaxR);
    return FVector(
        Center.X + FMath::Cos(Angle) * Radius,
        Center.Y + FMath::Sin(Angle) * Radius,
        0.0f);
}

UStaticMesh* LoadTargetLikeMesh(const FMVTrackJobManifest& Manifest)
{
    if (Manifest.TargetMeshPath.StartsWith(TEXT("/Game/")))
    {
        if (UStaticMesh* Mesh = LoadObject<UStaticMesh>(UStaticMesh::StaticClass(), *Manifest.TargetMeshPath))
        {
            return Mesh;
        }
    }
    if (!Manifest.TargetCategory.IsEmpty())
    {
        const FString AssetPath = FString::Printf(TEXT("/Game/Assets/Realistic/%s/model"), *Manifest.TargetCategory);
        if (UStaticMesh* Mesh = LoadObject<UStaticMesh>(UStaticMesh::StaticClass(), *AssetPath))
        {
            return Mesh;
        }
    }
    return nullptr;
}
}

AMVTrackSequenceRunner::AMVTrackSequenceRunner()
{
    PrimaryActorTick.bCanEverTick = true;
    PrimaryActorTick.TickInterval = 0.0f;
}

void AMVTrackSequenceRunner::BeginPlay()
{
    Super::BeginPlay();
    StartTimeSeconds = FPlatformTime::Seconds();
    JobConfig = NewObject<UMVTrackJobConfig>(this);
    CameraManager = NewObject<UMVTrackCameraManager>(this);
    TargetController = NewObject<UMVTrackTargetController>(this);
    AnnotationWriter = NewObject<UMVTrackAnnotationWriter>(this);
    RenderWriter = NewObject<UMVTrackRenderWriter>(this);
    CameraManager->RegisterComponent();
    TargetController->RegisterComponent();
    AnnotationWriter->RegisterComponent();
    RenderWriter->RegisterComponent();
    InitializeFromJob();
}

void AMVTrackSequenceRunner::InitializeFromJob()
{
    FString JobPath;
    FParse::Value(FCommandLine::Get(), TEXT("MVTrackJob="), JobPath);
    if (JobPath.IsEmpty() || !FPaths::FileExists(JobPath))
    {
        JobConfig->Manifest.JobId = TEXT("probe_001");
        JobConfig->Manifest.SequenceId = TEXT("seq_probe");
        JobConfig->Manifest.Seed = 42;
        JobConfig->Manifest.TargetCategory = TEXT("backpack");
        JobConfig->Manifest.TargetScaleM = 0.8f;
        JobConfig->Manifest.NumCameras = 3;
        JobConfig->Manifest.NumFrames = 30;
        JobConfig->Manifest.FPS = 30;
        JobConfig->Manifest.ResolutionX = 1280;
        JobConfig->Manifest.ResolutionY = 720;
        JobConfig->Manifest.NumOccluders = 3;
        JobConfig->Manifest.OutputDir = TEXT("/tmp/mvtrack_probe");
        JobConfig->Manifest.MotionType = TEXT("physics_impulse");
        JobConfig->bValid = true;
    }
    else
    {
        FString JsonStr;
        FFileHelper::LoadFileToString(JsonStr, *JobPath);
        TSharedPtr<FJsonObject> Root;
        TSharedRef<TJsonReader<>> Reader = TJsonReaderFactory<>::Create(JsonStr);
        if (FJsonSerializer::Deserialize(Reader, Root) && Root.IsValid())
        {
            JobConfig->Manifest.JobId = Root->GetStringField(TEXT("job_id"));
            JobConfig->Manifest.SequenceId = Root->GetStringField(TEXT("sequence_id"));
            JobConfig->Manifest.Seed = Root->GetIntegerField(TEXT("seed"));
            JobConfig->Manifest.TargetCategory = Root->GetStringField(TEXT("target_category"));
            Root->TryGetStringField(TEXT("target_mesh"), JobConfig->Manifest.TargetMeshPath);
            double GroundZ = 0.0;
            if (Root->TryGetNumberField(TEXT("target_ground_z_cm"), GroundZ))
            {
                JobConfig->Manifest.TargetGroundZCm = (float)GroundZ;
            }
            JobConfig->Manifest.TargetScaleM = (float)Root->GetNumberField(TEXT("target_scale_m"));
            JobConfig->Manifest.NumCameras = Root->GetIntegerField(TEXT("num_cameras"));
            JobConfig->Manifest.NumFrames = Root->GetIntegerField(TEXT("num_frames"));
            JobConfig->Manifest.FPS = Root->GetIntegerField(TEXT("fps"));
            JobConfig->Manifest.NumOccluders = Root->GetIntegerField(TEXT("num_occluders"));
            JobConfig->Manifest.OutputDir = Root->GetStringField(TEXT("output_dir"));
            JobConfig->Manifest.MotionType = Root->GetStringField(TEXT("target_motion_type"));
            const TArray<TSharedPtr<FJsonValue>>* ResArr;
            if (Root->TryGetArrayField(TEXT("resolution"), ResArr) && ResArr->Num() >= 2)
            {
                JobConfig->Manifest.ResolutionX = (*ResArr)[0]->AsNumber();
                JobConfig->Manifest.ResolutionY = (*ResArr)[1]->AsNumber();
            }
            JobConfig->bValid = true;
            SeqLog(FString::Printf(TEXT("Loaded: %s"), *JobConfig->Manifest.JobId));
        }
        else { SeqLog(TEXT("Parse failed")); return; }
    }
    const FMVTrackJobManifest& M = JobConfig->Manifest;
    RNG = FRandomStream(M.Seed);
    TotalFrames = M.NumFrames;
    FrameTime = 1.0f / (float)M.FPS;
    RenderWriter->InitRenderTargets(M.ResolutionX, M.ResolutionY);

    BuildRoomScene();

    // Spawn target in CENTER of room
    FVector StartPos(0, 0, 50);
    if (!TargetController->SpawnTarget(M, StartPos, RNG))
    {
        WriteSequenceManifest(EMVTrackFailure::AssetImportFailed);
        bFailed = true;
        return;
    }
    TargetController->RunSettlement(0.2f, 0.02f, 2.0f);

    FVector TargetPos = TargetController->GetTargetTransform().GetLocation();
    CameraManager->SetupCameras(M, TargetPos, TargetPos, RNG);
    SpawnRoomOccluders(TargetPos);
    CameraManager->SaveCalibrations(M.OutputDir);

    bRunning = true;
    SeqLog(FString::Printf(TEXT("Room ready. Target at (%.0f,%.0f,%.0f)"),
        TargetPos.X, TargetPos.Y, TargetPos.Z));
}

void AMVTrackSequenceRunner::BuildRoomScene()
{
    const int32 SceneStyleSeed = JobConfig ? JobConfig->Manifest.Seed : 1337;
    FRandomStream SceneRNG(SceneStyleSeed + 17);

    // Remove template-map meshes/lights so every generated sequence owns its scene.
    for (TActorIterator<AActor> It(GetWorld()); It; ++It)
    {
        AActor* Actor = *It;
        if (Actor && Actor != this && Actor->FindComponentByClass<UStaticMeshComponent>())
        {
            Actor->Destroy();
        }
    }
    for (TActorIterator<ADirectionalLight> It(GetWorld()); It; ++It)
    {
        It->Destroy();
    }
    for (TActorIterator<ASkyLight> It(GetWorld()); It; ++It)
    {
        It->Destroy();
    }
    for (TActorIterator<AExponentialHeightFog> It(GetWorld()); It; ++It)
    {
        It->Destroy();
    }

    FActorSpawnParameters P;
    P.SpawnCollisionHandlingOverride = ESpawnActorCollisionHandlingMethod::AlwaysSpawn;

    // Lighting: vary warmth and direction per sequence while keeping all views synchronized.
    ADirectionalLight* Sun = GetWorld()->SpawnActor<ADirectionalLight>(
        FVector(0, 0, 900),
        FRotator(SceneRNG.FRandRange(-68.0f, -38.0f), SceneRNG.FRandRange(-35.0f, 55.0f), 0),
        P);
    if (Sun)
    {
        UDirectionalLightComponent* C = Cast<UDirectionalLightComponent>(Sun->GetLightComponent());
        if (C)
        {
            C->SetIntensity(SceneRNG.FRandRange(4.0f, 10.0f));
            C->SetLightColor(FLinearColor(
                SceneRNG.FRandRange(0.92f, 1.0f),
                SceneRNG.FRandRange(0.88f, 0.98f),
                SceneRNG.FRandRange(0.78f, 0.92f)));
            C->SetCastShadows(true);
        }
    }
    ASkyLight* Sky = GetWorld()->SpawnActor<ASkyLight>(FVector(0,0,500), FRotator::ZeroRotator, P);
    if (Sky) { USkyLightComponent* C = Sky->GetLightComponent(); C->SetIntensity(SceneRNG.FRandRange(0.6f, 1.8f)); }
    AExponentialHeightFog* Fog = GetWorld()->SpawnActor<AExponentialHeightFog>(FVector::ZeroVector, FRotator::ZeroRotator, P);
    if (Fog) { UExponentialHeightFogComponent* C = Fog->GetComponent(); C->SetFogDensity(SceneRNG.FRandRange(0.0005f, 0.004f)); }

    auto MakePointLight = [&](FVector Pos, float Intensity)
    {
        APointLight* L = GetWorld()->SpawnActor<APointLight>(Pos, FRotator::ZeroRotator, P);
        if (!L) return;
        UPointLightComponent* C = Cast<UPointLightComponent>(
            L->GetComponentByClass(UPointLightComponent::StaticClass()));
        if (!C) return;
        C->SetIntensity(Intensity);
        C->SetAttenuationRadius(900.0f);
        C->SetLightColor(FLinearColor(1.0f, 0.94f, 0.86f));
        C->SetCastShadows(false);
    };

    MakePointLight(FVector(0, 0, 260), SceneRNG.FRandRange(2800.0f, 5200.0f));
    MakePointLight(FVector(250, 250, 230), SceneRNG.FRandRange(900.0f, 2600.0f));
    MakePointLight(FVector(-250, -250, 230), SceneRNG.FRandRange(900.0f, 2600.0f));

    auto MakeFloor = [&](FVector Pos, FVector Scale)
    {
        AStaticMeshActor* A = GetWorld()->SpawnActor<AStaticMeshActor>(Pos, FRotator::ZeroRotator, P);
        if (!A) return;
        UStaticMeshComponent* M = A->GetStaticMeshComponent();
        UStaticMesh* Plane = LoadObject<UStaticMesh>(UStaticMesh::StaticClass(), TEXT("/Engine/BasicShapes/Plane"));
        if (Plane) { M->SetStaticMesh(Plane); M->SetWorldScale3D(Scale); M->SetCollisionProfileName(TEXT("BlockAll")); }
        MVTrackMaterials::ApplySemanticMaterial(
            M, A, TEXT("warm_wood_floor"), SceneStyleSeed + 11,
            MVTrackMaterials::EMaterialRole::Floor);
    };

    auto MakeWall = [&](FVector Pos, FRotator Rot, FVector Scale)
    {
        AStaticMeshActor* A = GetWorld()->SpawnActor<AStaticMeshActor>(Pos, Rot, P);
        if (!A) return;
        UStaticMeshComponent* M = A->GetStaticMeshComponent();
        UStaticMesh* Cube = LoadObject<UStaticMesh>(UStaticMesh::StaticClass(), TEXT("/Engine/BasicShapes/Cube"));
        if (Cube) { M->SetStaticMesh(Cube); M->SetWorldScale3D(Scale); M->SetCollisionProfileName(TEXT("BlockAll")); }
        MVTrackMaterials::ApplySemanticMaterial(
            M, A, TEXT("painted_indoor_wall"), SceneStyleSeed + 29,
            MVTrackMaterials::EMaterialRole::Wall);
    };

    float R = SceneRNG.FRandRange(620.0f, 860.0f);
    float WT = 0.2f;
    float WH = SceneRNG.FRandRange(280.0f, 380.0f);

    MakeFloor(FVector(0, 0, 0), FVector(R/50.0f, R/50.0f, 1.0f));
    // Leave the ceiling open for off-screen captures; indoor lights provide the visible illumination.
    MakeWall(FVector(0, R, WH/2), FRotator(0,0,0), FVector(R/50.0f, WT, WH/50.0f));
    MakeWall(FVector(0, -R, WH/2), FRotator(0,0,0), FVector(R/50.0f, WT, WH/50.0f));
    MakeWall(FVector(-R, 0, WH/2), FRotator(0,90,0), FVector(R/50.0f, WT, WH/50.0f));
    MakeWall(FVector(R, 0, WH/2), FRotator(0,90,0), FVector(R/50.0f, WT, WH/50.0f));

    int32 FurnitureStyleIndex = 0;
    auto PlaceF = [&](const TCHAR* Path, FVector Pos, float S, float Yaw)
    {
        UStaticMesh* Mesh = LoadObject<UStaticMesh>(UStaticMesh::StaticClass(), Path);
        if (!Mesh) return;
        AStaticMeshActor* A = GetWorld()->SpawnActor<AStaticMeshActor>(Pos, FRotator(0, Yaw, 0), P);
        if (!A) return;
        UStaticMeshComponent* M = A->GetStaticMeshComponent();
        M->SetStaticMesh(Mesh); M->SetWorldScale3D(FVector(S)); M->SetCollisionProfileName(TEXT("BlockAll"));
        MVTrackMaterials::ApplySemanticMaterial(
            M, A, FString(Path), SceneStyleSeed + 100 + FurnitureStyleIndex++,
            MVTrackMaterials::EMaterialRole::Furniture);
    };

    PlaceF(FurniturePaths[2], FVector(-0.52f * R, 0.62f * R, 0), SceneRNG.FRandRange(0.9f, 1.35f), SceneRNG.FRandRange(-20.0f, 20.0f));
    PlaceF(FurniturePaths[6], FVector(0.48f * R, 0.58f * R, 0), SceneRNG.FRandRange(0.8f, 1.15f), SceneRNG.FRandRange(-12.0f, 12.0f));
    PlaceF(FurniturePaths[3], FVector(0, -0.50f * R, 0), SceneRNG.FRandRange(0.75f, 1.25f), SceneRNG.FRandRange(-30.0f, 30.0f));
    PlaceF(FurniturePaths[4], FVector(-0.62f * R, -0.35f * R, 0), SceneRNG.FRandRange(0.9f, 1.45f), SceneRNG.FRandRange(60.0f, 120.0f));

    const int32 ExtraFurniture = SceneRNG.RandRange(5, 11);
    for (int32 i = 0; i < ExtraFurniture; ++i)
    {
        const TCHAR* AssetPath = FurniturePaths[SceneRNG.RandRange(0, NumFurniture - 1)];
        FVector Pos = GroundLocationNear(FVector::ZeroVector, SceneRNG, 260.0f, FMath::Max(300.0f, R - 80.0f));
        Pos.X = FMath::Clamp(Pos.X, -R + 80.0f, R - 80.0f);
        Pos.Y = FMath::Clamp(Pos.Y, -R + 80.0f, R - 80.0f);
        PlaceF(AssetPath, Pos, SceneRNG.FRandRange(0.55f, 1.55f), SceneRNG.FRandRange(0.0f, 360.0f));
    }

    SeqLog(FString::Printf(TEXT("Room: varied %.1fm with %d extra furniture props"), R / 50.0f, ExtraFurniture));
}

void AMVTrackSequenceRunner::SpawnRoomOccluders(const FVector& TargetPos)
{
    FActorSpawnParameters P;
    P.SpawnCollisionHandlingOverride = ESpawnActorCollisionHandlingMethod::AlwaysSpawn;
    const FMVTrackJobManifest& MJob = JobConfig->Manifest;
    FRandomStream OccRNG(MJob.Seed + 503);
    const TArray<FMVTrackCameraData>& Cams = CameraManager->GetCameras();

    auto SpawnPrimitiveOccluder = [&](const FString& Name, UStaticMesh* Mesh, const FVector& Pos,
                                      const FRotator& Rot, const FVector& Scale, int32 SeedOffset)
    {
        AStaticMeshActor* Actor = GetWorld()->SpawnActor<AStaticMeshActor>(Pos, Rot, P);
        if (!Actor) return;
        UStaticMeshComponent* MC = Actor->GetStaticMeshComponent();
        if (Mesh) { MC->SetStaticMesh(Mesh); }
        MC->SetWorldScale3D(Scale);
        MC->SetCollisionProfileName(TEXT("BlockAll"));
        MVTrackMaterials::ApplySemanticMaterial(
            MC, Actor, Name, MJob.Seed + SeedOffset,
            MVTrackMaterials::EMaterialRole::Occluder);
    };

    UStaticMesh* Cube = LoadObject<UStaticMesh>(UStaticMesh::StaticClass(), TEXT("/Engine/BasicShapes/Cube"));
    UStaticMesh* Cyl = LoadObject<UStaticMesh>(UStaticMesh::StaticClass(), TEXT("/Engine/BasicShapes/Cylinder"));

    const int32 ViewSpecificOccluders = FMath::Clamp(MJob.NumOccluders / 2, 0, FMath::Min(2, Cams.Num()));
    for (int32 i = 0; i < ViewSpecificOccluders; ++i)
    {
        if (!Cams.IsValidIndex(i) || !Cams[i].CameraActor) continue;
        const FVector CamPos = Cams[i].CameraActor->GetActorLocation();
        const FVector Ray = TargetPos - CamPos;
        const FVector Dir = Ray.GetSafeNormal();
        FVector Right = FVector::CrossProduct(FVector::UpVector, Dir).GetSafeNormal();
        if (Right.IsNearlyZero()) { Right = FVector::RightVector; }

        const float DistFrac = OccRNG.FRandRange(0.42f, 0.62f);
        const float Side = (i % 2 == 0) ? 1.0f : -1.0f;
        FVector Pos = CamPos + Ray * DistFrac + Right * Side * OccRNG.FRandRange(45.0f, 95.0f);
        Pos.Z = FMath::Clamp(TargetPos.Z + OccRNG.FRandRange(-8.0f, 40.0f), 25.0f, 135.0f);

        const bool bPillar = (i % 2 == 0);
        const FVector Scale = bPillar
            ? FVector(OccRNG.FRandRange(0.04f, 0.09f), OccRNG.FRandRange(0.04f, 0.09f), OccRNG.FRandRange(0.55f, 1.25f))
            : FVector(OccRNG.FRandRange(0.18f, 0.48f), OccRNG.FRandRange(0.04f, 0.11f), OccRNG.FRandRange(0.30f, 0.85f));
        SpawnPrimitiveOccluder(
            bPillar ? TEXT("view_specific_pillar") : TEXT("view_specific_panel"),
            bPillar ? Cyl : Cube,
            Pos,
            FRotator(0, OccRNG.FRandRange(0.0f, 360.0f), 0),
            Scale,
            700 + i * 11);
    }

    const int32 NearOccluders = FMath::Clamp(MJob.NumOccluders - ViewSpecificOccluders, 0, 2);
    for (int32 i = 0; i < NearOccluders; ++i)
    {
        FVector Pos = GroundLocationNear(TargetPos, OccRNG, 140.0f, 260.0f);
        Pos.Z = OccRNG.FRandRange(18.0f, 60.0f);
        SpawnPrimitiveOccluder(
            TEXT("near_target_clutter_occluder"),
            Cube,
            Pos,
            FRotator(0, OccRNG.FRandRange(0.0f, 360.0f), 0),
            FVector(OccRNG.FRandRange(0.18f, 0.62f), OccRNG.FRandRange(0.08f, 0.28f), OccRNG.FRandRange(0.20f, 0.72f)),
            820 + i * 13);
    }

    // Similar distractors are intentionally close to the target category. They teach
    // identity preservation instead of merely foreground/background separation.
    UStaticMesh* TargetLike = LoadTargetLikeMesh(MJob);
    const int32 SimilarDistractors = OccRNG.RandRange(1, 3);
    for (int32 i = 0; i < SimilarDistractors; ++i)
    {
        UStaticMesh* Mesh = TargetLike ? TargetLike : LoadObject<UStaticMesh>(UStaticMesh::StaticClass(), FurniturePaths[OccRNG.RandRange(0, NumFurniture - 1)]);
        if (!Mesh) continue;
        FVector Pos = GroundLocationNear(TargetPos, OccRNG, 210.0f, 360.0f);
        Pos.Z = TargetPos.Z;
        AStaticMeshActor* A = GetWorld()->SpawnActor<AStaticMeshActor>(
            Pos, FRotator(0, OccRNG.FRandRange(0.0f, 360.0f), 0), P);
        if (!A) continue;
        UStaticMeshComponent* MC = A->GetStaticMeshComponent();
        MC->SetStaticMesh(Mesh);
        MC->SetWorldScale3D(FVector(OccRNG.FRandRange(0.58f, 0.98f)));
        MC->SetCollisionProfileName(TEXT("BlockAll"));
    }

    SeqLog(FString::Printf(TEXT("Hard scene: %d view occluders, %d near occluders, %d similar distractors"),
        ViewSpecificOccluders, NearOccluders, SimilarDistractors));
}

void AMVTrackSequenceRunner::ApplyOccluderMaterials() {}

void AMVTrackSequenceRunner::Tick(float DeltaTime)
{
    Super::Tick(DeltaTime);
    if (!bRunning || bFailed) return;
    if (CurrentFrame >= TotalFrames)
    {
        WriteSequenceManifest(EMVTrackFailure::None);
        bRunning = false;
        return;
    }
    RunFrame();
    CurrentFrame++;
    SimulationTime += FrameTime;
}

void AMVTrackSequenceRunner::RunFrame()
{
    const FMVTrackJobManifest& M = JobConfig->Manifest;

    // Move target along a non-circular path so fixed cameras see clear
    // horizontal and depth displacement instead of a simple orbit.
    float T = (float)CurrentFrame / (float)TotalFrames;
    float Angle = T * 2.0f * PI;
    float PathX = FMath::Sin(Angle * 1.35f + 0.35f) * 150.0f
        + FMath::Sin(Angle * 2.7f) * 35.0f;
    float PathY = FMath::Sin(Angle * 0.85f + 1.1f) * 115.0f
        + FMath::Cos(Angle * 2.1f) * 28.0f;

    if (TargetController->TargetActor)
    {
        float GroundZ = FMath::Abs(M.TargetGroundZCm) > 0.01f
            ? M.TargetGroundZCm
            : TargetController->TargetActor->GetActorLocation().Z;
        FVector Goal(PathX, PathY, GroundZ);
        TargetController->TargetActor->SetActorLocation(Goal, false);
        TargetController->TargetActor->SetActorRotation(FRotator(
            0.0f,
            Angle * 57.2958f + FMath::Sin(Angle * 1.7f) * 24.0f,
            0.0f));
        if (CurrentFrame == 0)
        {
            SeqLog(FString::Printf(TEXT("Frame0 forced target loc=%s"),
                *TargetController->TargetActor->GetActorLocation().ToString()));
        }
    }

    FVector TargetPos = TargetController->GetTargetTransform().GetLocation();
    // Keep cameras fixed after setup. Multi-view SOT should make the object move
    // through each view instead of letting virtual cameras track it every frame.

    TArray<bool> Occluded = CameraManager->CheckOcclusion(TargetPos, 20.0f);
    TArray<FMVTrackMaskStats> MaskStats;
    MaskStats.SetNum(M.NumCameras);

    for (int32 i = 0; i < M.NumCameras; i++)
    {
        FString CamId = FString::Printf(TEXT("cam_%03d"), i);
        FString FrameDir = FString::Printf(TEXT("%s/frames/%s"), *M.OutputDir, *CamId);
        USceneCaptureComponent2D* CC = CameraManager->GetCaptureComp(i);
        RenderWriter->CaptureAll(CC, TargetController->TargetActor, FrameDir, CurrentFrame, &MaskStats[i]);
    }

    auto Calibs = CameraManager->GetAllCalibrations();
    FTransform TX = TargetController->GetTargetTransform();
    FBox TB = TargetController->GetTargetWorldBounds();
    float TS = CurrentFrame * FrameTime;
    for (int32 i = 0; i < M.NumCameras; i++)
    {
        FString CamId = FString::Printf(TEXT("cam_%03d"), i);
        if (i >= Calibs.Num()) continue;
        FMVTrackFrameAnnotation Ann = AnnotationWriter->GenerateAnnotationFromMask(
            CurrentFrame, TS, CamId, Calibs[i].CameraToWorld,
            2.0f * FMath::RadiansToDegrees(FMath::Atan(18.0f / Calibs[i].FocalLengthMM)),
            M.ResolutionX, M.ResolutionY, TX, TB, 1, M.TargetCategory,
            Occluded.IsValidIndex(i) ? Occluded[i] : false,
            MaskStats.IsValidIndex(i) ? MaskStats[i] : FMVTrackMaskStats());
        AllAnnotations.Add(Ann);
        FString AP = FString::Printf(TEXT("%s/frames/%s/ann/%06d.json"), *M.OutputDir, *CamId, CurrentFrame);
        UMVTrackAnnotationWriter::SaveAnnotationJson(Ann, AP);
    }

    if (CurrentFrame % 5 == 0)
        SeqLog(FString::Printf(TEXT("Frame %d/%d target=(%.0f,%.0f,%.0f)"),
            CurrentFrame, TotalFrames, TargetPos.X, TargetPos.Y, TargetPos.Z));
}

bool AMVTrackSequenceRunner::OnlineQualityCheck()
{
    if (!TargetController || !TargetController->TargetActor) return false;
    FVector L = TargetController->GetTargetTransform().GetLocation();
    return !L.ContainsNaN() && L.Size() < 10000.0f && L.Z > -100.0f;
}

void AMVTrackSequenceRunner::SetupEnvironment() {}
void AMVTrackSequenceRunner::SpawnFloor() {}

void AMVTrackSequenceRunner::WriteSequenceManifest(EMVTrackFailure Failure)
{
    const FMVTrackJobManifest& M = JobConfig->Manifest;
    TSharedPtr<FJsonObject> MF = MakeShareable(new FJsonObject);
    MF->SetStringField(TEXT("job_id"), M.JobId);
    MF->SetStringField(TEXT("sequence_id"), M.SequenceId);
    MF->SetNumberField(TEXT("seed"), M.Seed);
    MF->SetStringField(TEXT("target_category"), M.TargetCategory);
    MF->SetNumberField(TEXT("num_cameras"), M.NumCameras);
    MF->SetNumberField(TEXT("num_frames"), CurrentFrame);
    MF->SetNumberField(TEXT("fps"), M.FPS);
    MF->SetNumberField(TEXT("generation_time_sec"), FPlatformTime::Seconds() - StartTimeSeconds);
    MF->SetBoolField(TEXT("success"), Failure == EMVTrackFailure::None);
    MF->SetNumberField(TEXT("total_annotations"), AllAnnotations.Num());
    FString RD = M.OutputDir / TEXT("frames/cam_000/rgb");
    TArray<FString> RF;
    IFileManager::Get().FindFiles(RF, *RD, TEXT("*.png"));
    MF->SetNumberField(TEXT("rendered_rgb_frames"), RF.Num());
    FString JS;
    TSharedRef<TJsonWriter<>> W = TJsonWriterFactory<>::Create(&JS);
    FJsonSerializer::Serialize(MF.ToSharedRef(), W);
    FFileHelper::SaveStringToFile(JS, *(M.OutputDir / TEXT("seq_meta.json")));
    SeqLog(FString::Printf(TEXT("Done: %s (%.1fs, %d anns, %d rgb)"),
        *M.JobId, FPlatformTime::Seconds() - StartTimeSeconds, AllAnnotations.Num(), RF.Num()));
    TargetController->Cleanup();
    RenderWriter->Cleanup();
    UKismetSystemLibrary::QuitGame(this, nullptr, EQuitPreference::Quit, false);
}

void AMVTrackSequenceRunner::SeqLog(const FString& Msg)
{
    UE_LOG(LogTemp, Log, TEXT("[MVTrack:%s] %s"), *JobConfig->Manifest.SequenceId, *Msg);
}
