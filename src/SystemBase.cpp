#include "SystemBase.h"

#include <stdexcept>

void SystemBase::setMaxIntegrationStep(double h)
{
    if (!(h > 0.0))
    {
        throw std::invalid_argument("SystemBase::setMaxIntegrationStep: step must be positive");
    }
    maxIntegrationStep_ = h;
}
