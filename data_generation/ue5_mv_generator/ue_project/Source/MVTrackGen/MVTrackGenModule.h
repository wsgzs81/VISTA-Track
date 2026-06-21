// MVTrackGenModule.h — Main game module
#pragma once

#include "CoreMinimal.h"
#include "Modules/ModuleManager.h"

class FMVTrackGenModule : public IModuleInterface
{
public:
    virtual void StartupModule() override;
    virtual void ShutdownModule() override;
};
