#include "MVTrackTargetController.h"
#include "MVTrackMaterialLibrary.h"
#include "Components/StaticMeshComponent.h"
#include "Engine/StaticMeshActor.h"
#include "Engine/StaticMesh.h"
#include "Engine/World.h"
#include "Engine/OverlapResult.h"
#include "Materials/MaterialInstanceDynamic.h"
#include "Components/SceneComponent.h"

static const TCHAR* RealisticPaths[] = {
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
static const int32 NumRealPaths = 9;

bool UMVTrackTargetController::SpawnTarget(
    const FMVTrackJobManifest& Manifest,
    const FVector& SpawnLocation,
    FRandomStream& RNG)
{
    float ScaleM = Manifest.TargetScaleM;

    FActorSpawnParameters P;
    P.SpawnCollisionHandlingOverride = ESpawnActorCollisionHandlingMethod::AlwaysSpawn;
    AStaticMeshActor* MeshActor = GetWorld()->SpawnActor<AStaticMeshActor>(
        SpawnLocation,
        FRotator(0.0f, RNG.FRandRange(0.0f, 360.0f), 0.0f),
        P);
    if (!MeshActor) return false;

    UStaticMeshComponent* MeshComp = MeshActor->GetStaticMeshComponent();
    MeshActor->SetMobility(EComponentMobility::Movable);
    MeshComp->SetMobility(EComponentMobility::Movable);
    UStaticMesh* Mesh = nullptr;

    FString MeshKey = Manifest.TargetMeshPath;
    if (MeshKey.IsEmpty())
    {
        MeshKey = Manifest.TargetCategory;
    }

    if (MeshKey.Contains(TEXT("sphere")))
    {
        Mesh = LoadObject<UStaticMesh>(UStaticMesh::StaticClass(), TEXT("/Engine/BasicShapes/Sphere"));
    }
    else if (MeshKey.Contains(TEXT("cylinder")))
    {
        Mesh = LoadObject<UStaticMesh>(UStaticMesh::StaticClass(), TEXT("/Engine/BasicShapes/Cylinder"));
    }
    else if (MeshKey.Contains(TEXT("cone")))
    {
        Mesh = LoadObject<UStaticMesh>(UStaticMesh::StaticClass(), TEXT("/Engine/BasicShapes/Cone"));
    }
    else if (MeshKey.Contains(TEXT("torus")))
    {
        Mesh = LoadObject<UStaticMesh>(UStaticMesh::StaticClass(), TEXT("/Engine/EditorMeshes/EditorTorus"));
    }

    if (!Mesh && MeshKey.StartsWith(TEXT("/Game/")))
    {
        Mesh = LoadObject<UStaticMesh>(UStaticMesh::StaticClass(), *MeshKey);
    }
    if (!Mesh && Manifest.TargetCategory.Len() > 0)
    {
        const FString AssetPath = FString::Printf(TEXT("/Game/Assets/Realistic/%s/model"), *Manifest.TargetCategory);
        Mesh = LoadObject<UStaticMesh>(UStaticMesh::StaticClass(), *AssetPath);
    }
    if (!Mesh)
    {
        Mesh = LoadObject<UStaticMesh>(UStaticMesh::StaticClass(), TEXT("/Engine/BasicShapes/Cube"));
    }
    if (Mesh) MeshComp->SetStaticMesh(Mesh);

    FVector BoundsExtent = Mesh ? Mesh->GetBounds().BoxExtent : FVector(50.0f);
    float MaxExtent = FMath::Max3(BoundsExtent.X, BoundsExtent.Y, BoundsExtent.Z);
    float DesiredHalfExtentCm = FMath::Max(ScaleM * 100.0f * 0.5f, 12.0f);
    float UniformScale = MaxExtent > 1.0f ? DesiredHalfExtentCm / MaxExtent : ScaleM;
    MeshComp->SetWorldScale3D(FVector(UniformScale));

    MeshComp->SetSimulatePhysics(false);
    MeshComp->SetEnableGravity(false);
    MeshComp->SetLinearDamping(0.3f);
    MeshComp->SetAngularDamping(0.5f);
    MeshComp->SetCollisionProfileName(TEXT("BlockAll"));

    const bool bIsBuiltinFallback =
        MeshKey.StartsWith(TEXT("builtin_")) ||
        MeshKey.StartsWith(TEXT("/Engine/")) ||
        MeshKey.Contains(TEXT("sphere")) ||
        MeshKey.Contains(TEXT("cylinder")) ||
        MeshKey.Contains(TEXT("cone")) ||
        MeshKey.Contains(TEXT("torus"));
    if (bIsBuiltinFallback)
    {
        MVTrackMaterials::ApplySemanticMaterial(
            MeshComp,
            MeshActor,
            Manifest.TargetCategory.IsEmpty() ? MeshKey : Manifest.TargetCategory,
            Manifest.Seed + 101,
            MVTrackMaterials::EMaterialRole::Target);
    }

    TargetActor = MeshActor;
    TargetMesh = MeshComp;

    FVector GroundedLocation = MeshActor->GetActorLocation();
    if (FMath::Abs(Manifest.TargetGroundZCm) > 0.01f)
    {
        GroundedLocation.Z = Manifest.TargetGroundZCm;
    }
    else if (Mesh)
    {
        const FBoxSphereBounds LocalBounds = Mesh->GetBounds();
        GroundedLocation.Z = 2.0f - LocalBounds.Origin.Z * UniformScale + LocalBounds.BoxExtent.Z * UniformScale;
    }
    else
    {
        GroundedLocation.Z = SpawnLocation.Z;
    }
    MeshActor->SetActorLocation(GroundedLocation, false);
    MeshComp->UpdateBounds();

    UE_LOG(LogTemp, Log, TEXT("[MVTrack] Spawned target %s mesh=%s at %s grounded=%s scale=%.3f"),
        *Manifest.TargetCategory, *MeshKey, *MeshActor->GetActorLocation().ToString(), UniformScale);
    return true;
}

bool UMVTrackTargetController::SpawnOccluders(
    const FMVTrackJobManifest& Manifest,
    const FVector& TargetLocation,
    const TArray<FVector>& CameraLocations,
    FRandomStream& RNG)
{
    for (int32 i = 0; i < Manifest.NumOccluders; i++)
    {
        FString OccType = (i % 2 == 0) ? TEXT("pillar") : TEXT("wall");
        float OccScaleM = RNG.FRandRange(0.5f, 2.0f);
        FVector Offset(RNG.FRandRange(-300, 300), RNG.FRandRange(-300, 300), 0);
        FVector OccPos = TargetLocation + Offset;
        OccPos.Z = TargetLocation.Z;

        FVector OccScale;
        if (OccType == TEXT("pillar"))
            OccScale = FVector(OccScaleM * 50, OccScaleM * 50, RNG.FRandRange(200, 400));
        else
            OccScale = FVector(OccScaleM * 100, RNG.FRandRange(20, 50), RNG.FRandRange(150, 300));

        FActorSpawnParameters P;
        P.SpawnCollisionHandlingOverride = ESpawnActorCollisionHandlingMethod::AlwaysSpawn;
        AStaticMeshActor* A = GetWorld()->SpawnActor<AStaticMeshActor>(
            OccPos, FRotator(0, RNG.FRandRange(0,360), 0), P);
        if (!A) continue;

        UStaticMeshComponent* MC = A->GetStaticMeshComponent();
        UStaticMesh* Mesh = LoadObject<UStaticMesh>(UStaticMesh::StaticClass(),
            OccType == TEXT("pillar") ? TEXT("/Engine/BasicShapes/Cylinder") : TEXT("/Engine/BasicShapes/Cube"));
        if (Mesh) MC->SetStaticMesh(Mesh);
        MC->SetWorldScale3D(OccScale / 100.0f);
        MC->SetCollisionProfileName(TEXT("BlockAll"));
        MVTrackMaterials::ApplySemanticMaterial(
            MC, A, OccType, Manifest.Seed + 3000 + i,
            MVTrackMaterials::EMaterialRole::Occluder);
        OccluderActors.Add(A);
    }

    // Background furniture
    for (int32 i = 0; i < RNG.RandRange(3, 6); i++)
    {
        const TCHAR* AssetPath = RealisticPaths[RNG.RandRange(0, NumRealPaths - 1)];
        UStaticMesh* Mesh = LoadObject<UStaticMesh>(UStaticMesh::StaticClass(), AssetPath);
        if (!Mesh) continue;
        FVector Pos(TargetLocation.X + RNG.FRandRange(-500, 500),
                    TargetLocation.Y + RNG.FRandRange(-500, 500), 0);
        FActorSpawnParameters P;
        P.SpawnCollisionHandlingOverride = ESpawnActorCollisionHandlingMethod::AlwaysSpawn;
        AStaticMeshActor* A = GetWorld()->SpawnActor<AStaticMeshActor>(Pos,
            FRotator(0, RNG.FRandRange(0,360), 0), P);
        if (!A) continue;
        UStaticMeshComponent* MC = A->GetStaticMeshComponent();
        MC->SetStaticMesh(Mesh);
        MC->SetWorldScale3D(FVector(RNG.FRandRange(0.8f, 2.0f)));
        MC->SetCollisionProfileName(TEXT("BlockAll"));
    }

    UE_LOG(LogTemp, Log, TEXT("[MVTrack] Spawned %d occluders + furniture"), OccluderActors.Num());
    return true;
}

bool UMVTrackTargetController::RunSettlement(float SettleTimeSec, float, float)
{
    if (!TargetMesh) return false;
    FPlatformProcess::Sleep(SettleTimeSec);
    return true;
}

bool UMVTrackTargetController::CheckPenetration() const { return false; }

void UMVTrackTargetController::ApplyMotion(int32 FrameIndex, FRandomStream& RNG)
{
    if (!TargetMesh) return;
    FVector Dir = FMath::VRand();
    Dir.Z = FMath::Abs(Dir.Z) * 0.5f;
    Dir.Normalize();
    TargetMesh->AddImpulse(Dir * RNG.FRandRange(100, 400), NAME_None, true);
}

FTransform UMVTrackTargetController::GetTargetTransform() const
{
    return TargetActor ? TargetActor->GetActorTransform() : FTransform::Identity;
}

FBox UMVTrackTargetController::GetTargetWorldBounds() const
{
    return TargetMesh ? TargetMesh->Bounds.GetBox() : FBox(ForceInit);
}

void UMVTrackTargetController::Cleanup()
{
    if (TargetActor) { TargetActor->Destroy(); TargetActor = nullptr; TargetMesh = nullptr; }
    for (AActor* A : OccluderActors) { if (A) A->Destroy(); }
    OccluderActors.Empty();
}

AActor* UMVTrackTargetController::SpawnPrimitiveMesh(
    const FString& MeshType, const FVector& Location, const FVector& Scale,
    bool bPhysicsEnabled, bool bIsDynamic, FRandomStream& RNG)
{
    FActorSpawnParameters P;
    P.SpawnCollisionHandlingOverride = ESpawnActorCollisionHandlingMethod::AlwaysSpawn;
    AStaticMeshActor* A = GetWorld()->SpawnActor<AStaticMeshActor>(Location,
        FRotator(RNG.FRandRange(0,360), RNG.FRandRange(0,360), RNG.FRandRange(0,360)), P);
    if (!A) return nullptr;
    UStaticMeshComponent* MC = A->GetStaticMeshComponent();
    UStaticMesh* Cube = LoadObject<UStaticMesh>(UStaticMesh::StaticClass(), TEXT("/Engine/BasicShapes/Cube"));
    if (Cube) MC->SetStaticMesh(Cube);
    MC->SetWorldScale3D(Scale / 100.0f);
    MVTrackMaterials::ApplySemanticMaterial(
        MC, A, MeshType, RNG.RandRange(10000, 999999),
        MVTrackMaterials::EMaterialRole::Prop);
    if (bPhysicsEnabled) { MC->SetSimulatePhysics(bIsDynamic); MC->SetEnableGravity(true); }
    else { MC->SetCollisionProfileName(TEXT("BlockAll")); }
    return A;
}

FVector UMVTrackTargetController::ComputeOccluderPosition(
    const FVector& CameraPos, const FVector& TargetPos,
    float DistanceFraction, float, FRandomStream&)
{
    FVector Dir = (TargetPos - CameraPos).GetSafeNormal();
    return CameraPos + Dir * (TargetPos - CameraPos).Size() * DistanceFraction;
}
