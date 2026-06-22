// MVTrackSequenceRunner.h — Room scene with tracking target
#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "MVTrackTypes.h"
#include "MVTrackSequenceRunner.generated.h"

class UMVTrackJobConfig;
class UMVTrackCameraManager;
class UMVTrackTargetController;
class UMVTrackAnnotationWriter;
class UMVTrackRenderWriter;
class AStaticMeshActor;

struct FMVTrackTransientOccluder
{
    AStaticMeshActor* Actor = nullptr;
    int32 CameraIndex = 0;
    float DistanceFraction = 0.45f;
    float StartT = 0.25f;
    float EndT = 0.65f;
    float TravelCm = 260.0f;
    float SideSign = 1.0f;
    float ZCenterCm = 85.0f;
};

UCLASS()
class MVTRACKRUNTIME_API AMVTrackSequenceRunner : public AActor
{
    GENERATED_BODY()

public:
    AMVTrackSequenceRunner();
    virtual void BeginPlay() override;
    virtual void Tick(float DeltaTime) override;

protected:
    void InitializeFromJob();
    void BuildRoomScene();
    void SpawnRoomOccluders(const FVector& TargetPos);
    void UpdateTransientOccluders(const FVector& TargetPos);
    void ApplyOccluderMaterials();
    void SetupEnvironment();
    void SpawnFloor();
    void RunFrame();
    void WriteSequenceManifest(EMVTrackFailure Failure = EMVTrackFailure::None);
    bool OnlineQualityCheck();
    void SeqLog(const FString& Msg);

    UPROPERTY() UMVTrackJobConfig* JobConfig = nullptr;
    UPROPERTY() UMVTrackCameraManager* CameraManager = nullptr;
    UPROPERTY() UMVTrackTargetController* TargetController = nullptr;
    UPROPERTY() UMVTrackAnnotationWriter* AnnotationWriter = nullptr;
    UPROPERTY() UMVTrackRenderWriter* RenderWriter = nullptr;

    int32 CurrentFrame = 0;
    int32 TotalFrames = 30;
    float FrameTime = 0.0f;
    float SimulationTime = 0.0f;
    bool bRunning = false;
    bool bFailed = false;
    bool bCaptureWarmupDone = false;
    FRandomStream RNG;
    TArray<FMVTrackTransientOccluder> TransientOccluders;
    TArray<FMVTrackFrameAnnotation> AllAnnotations;
    double StartTimeSeconds = 0.0;
};
