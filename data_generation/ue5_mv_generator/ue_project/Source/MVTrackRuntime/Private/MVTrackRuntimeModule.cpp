// MVTrackRuntimeModule.cpp
#include "MVTrackRuntimeModule.h"

#define LOCTEXT_NAMESPACE "FMVTrackRuntimeModule"

void FMVTrackRuntimeModule::StartupModule()
{
    UE_LOG(LogTemp, Log, TEXT("[MVTrack] Runtime module started"));
}

void FMVTrackRuntimeModule::ShutdownModule()
{
    UE_LOG(LogTemp, Log, TEXT("[MVTrack] Runtime module shutdown"));
}

#undef LOCTEXT_NAMESPACE

IMPLEMENT_MODULE(FMVTrackRuntimeModule, MVTrackRuntime)
