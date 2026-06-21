// MVTrackCameraManager.h — Multi-camera system with synchronized capture
#pragma once

#include "CoreMinimal.h"
#include "MVTrackTypes.h"
#include "MVTrackCameraManager.generated.h"

class USceneCaptureComponent2D;

/** Data for one capture camera in the array */
USTRUCT()
struct FMVTrackCameraData
{
    GENERATED_BODY()

    UPROPERTY() FString CameraId;
    UPROPERTY() AActor* CameraActor = nullptr;
    UPROPERTY() USceneCaptureComponent2D* CaptureComp = nullptr;
    UPROPERTY() FMVTrackCameraCalibration Calibration;
};

/**
 * Manages the multi-camera array: placement, synchronization, capture.
 * Cameras are placed on a hemisphere around the target trajectory center.
 */
UCLASS()
class MVTRACKRUNTIME_API UMVTrackCameraManager : public UActorComponent
{
    GENERATED_BODY()

public:
    /** Spawn N cameras on hemisphere around target center */
    void SetupCameras(const FMVTrackJobManifest& Manifest,
                      const FVector& TrajectoryCenter,
                      const FVector& TargetInitialPos,
                      FRandomStream& RNG);

    /** Point all cameras at the target location */
    void UpdateCameraLookAt(const FVector& TargetLocation);

    /** Capture all cameras for current frame (RGB, depth, mask) */
    void CaptureAllCameras(int32 FrameIndex, const FString& OutputDir);

    /** Get calibration data for all cameras */
    TArray<FMVTrackCameraCalibration> GetAllCalibrations() const;
    /** Get capture component for a specific camera */
    USceneCaptureComponent2D* GetCaptureComp(int32 Index) const;
    /** Get camera data array */
    const TArray<FMVTrackCameraData>& GetCameras() const { return Cameras; }

    /** Save camera calibration JSONs */
    void SaveCalibrations(const FString& OutputDir) const;

    /** Get number of cameras that can see the target (have line of sight) */
    int32 CountVisibleCameras(const FVector& TargetLocation, float TargetRadius) const;

    /** Check which cameras are occluded by occluders */
    TArray<bool> CheckOcclusion(const FVector& TargetLocation, float TargetRadius) const;

    UPROPERTY() int32 NumCameras = 4;

private:
    UPROPERTY() TArray<FMVTrackCameraData> Cameras;

    FVector PlaceCameraOnHemisphere(int32 Index, int32 Total,
                                     const FVector& Center, float Radius,
                                     float HeightMin, float HeightMax,
                                     float YawCoverageDeg,
                                     float PitchMinDeg, float PitchMaxDeg,
                                     FRandomStream& RNG);

    void ComputeCalibration(FMVTrackCameraData& CamData, int32 ResX, int32 ResY);
};
