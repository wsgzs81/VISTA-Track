// MVTrackGameMode.h — GameMode that spawns the SequenceRunner on BeginPlay
#pragma once

#include "CoreMinimal.h"
#include "GameFramework/GameModeBase.h"
#include "MVTrackGameMode.generated.h"

UCLASS()
class MVTRACKRUNTIME_API AMVTrackGameMode : public AGameModeBase
{
    GENERATED_BODY()

public:
    virtual void BeginPlay() override;
};
