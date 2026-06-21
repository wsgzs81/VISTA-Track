// MVTrackRenderWriter.h — RGB/Depth capture with Xvfb
#pragma once

#include "CoreMinimal.h"
#include "MVTrackRenderWriter.generated.h"

class USceneCaptureComponent2D;
class UTextureRenderTarget2D;

USTRUCT()
struct FMVTrackMaskStats
{
    GENERATED_BODY()

    UPROPERTY() bool bValid = false;
    UPROPERTY() int32 VisiblePixels = 0;
    UPROPERTY() FVector4f VisibleBBox = FVector4f::Zero();
};

UCLASS()
class MVTRACKRUNTIME_API UMVTrackRenderWriter : public UActorComponent
{
    GENERATED_BODY()

public:
    bool InitRenderTargets(int32 ResX, int32 ResY);
    bool CaptureRGB(USceneCaptureComponent2D* CaptureComp, const FString& OutputPath);
    bool CaptureDepth(USceneCaptureComponent2D* CaptureComp, const FString& OutputPath);
    FMVTrackMaskStats CaptureMask(USceneCaptureComponent2D* CaptureComp,
                                  AActor* TargetActor,
                                  const FString& OutputPath);
    bool CaptureAll(USceneCaptureComponent2D* CaptureComp,
                    AActor* TargetActor,
                    const FString& FrameDir,
                    int32 FrameIndex,
                    FMVTrackMaskStats* OutMaskStats = nullptr);
    void Cleanup();

private:
    UPROPERTY() UTextureRenderTarget2D* RGBTarget = nullptr;
    UPROPERTY() UTextureRenderTarget2D* DepthTarget = nullptr;
    UPROPERTY() UTextureRenderTarget2D* MaskTarget = nullptr;
    int32 Width = 1280;
    int32 Height = 720;
};
