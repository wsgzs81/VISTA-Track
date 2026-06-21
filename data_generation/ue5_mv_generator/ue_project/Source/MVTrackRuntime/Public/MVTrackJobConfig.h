// MVTrackJobConfig.h — Job manifest data holder
#pragma once

#include "CoreMinimal.h"
#include "MVTrackTypes.h"
#include "MVTrackJobConfig.generated.h"

/**
 * Simple data holder for job manifest.
 * Not a subsystem — created directly by SequenceRunner.
 */
UCLASS()
class MVTRACKRUNTIME_API UMVTrackJobConfig : public UActorComponent
{
    GENERATED_BODY()

public:
    UPROPERTY() FMVTrackJobManifest Manifest;
    UPROPERTY() bool bValid = false;
    UPROPERTY() FString JobFilePath;
};
