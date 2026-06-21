#include "MVTrackGameMode.h"
#include "MVTrackSequenceRunner.h"
#include "Engine/World.h"

void AMVTrackGameMode::BeginPlay()
{
    Super::BeginPlay();

    // Spawn the sequence runner actor
    FActorSpawnParameters Params;
    Params.SpawnCollisionHandlingOverride = ESpawnActorCollisionHandlingMethod::AlwaysSpawn;
    GetWorld()->SpawnActor<AMVTrackSequenceRunner>(FVector::ZeroVector, FRotator::ZeroRotator, Params);

    UE_LOG(LogTemp, Log, TEXT("[MVTrack] GameMode spawned SequenceRunner"));
}
