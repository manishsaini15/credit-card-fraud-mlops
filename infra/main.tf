# -----------------------------------------------------------------------------
# Resource Group
# -----------------------------------------------------------------------------

resource "azurerm_resource_group" "rg" {

  name     = var.resource_group_name

  location = var.location
}

# -----------------------------------------------------------------------------
# Storage Account
# -----------------------------------------------------------------------------

resource "azurerm_storage_account" "storage" {

  name                     = var.storage_account_name

  resource_group_name      = azurerm_resource_group.rg.name

  location                 = azurerm_resource_group.rg.location

  account_tier             = "Standard"

  account_replication_type = "LRS"
}

# -----------------------------------------------------------------------------
# Key Vault
# -----------------------------------------------------------------------------

resource "azurerm_key_vault" "kv" {

  name                = var.key_vault_name

  location            = azurerm_resource_group.rg.location

  resource_group_name = azurerm_resource_group.rg.name

  tenant_id = data.azurerm_client_config.current.tenant_id

  sku_name = "standard"

  purge_protection_enabled = false

  soft_delete_retention_days = 7
}

# -----------------------------------------------------------------------------
# Application Insights
# -----------------------------------------------------------------------------

resource "azurerm_application_insights" "appi" {

  name                = var.application_insights_name

  location            = azurerm_resource_group.rg.location

  resource_group_name = azurerm_resource_group.rg.name

  application_type = "web"
}

# -----------------------------------------------------------------------------
# Azure Container Registry
# -----------------------------------------------------------------------------

resource "azurerm_container_registry" "acr" {

  name                = var.container_registry_name

  location            = azurerm_resource_group.rg.location

  resource_group_name = azurerm_resource_group.rg.name

  sku = "Basic"

  admin_enabled = true
}

# -----------------------------------------------------------------------------
# Azure Machine Learning Workspace
# -----------------------------------------------------------------------------

resource "azurerm_machine_learning_workspace" "ml_workspace" {

  name                = var.workspace_name

  location            = azurerm_resource_group.rg.location

  resource_group_name = azurerm_resource_group.rg.name

  application_insights_id = azurerm_application_insights.appi.id

  key_vault_id = azurerm_key_vault.kv.id

  storage_account_id = azurerm_storage_account.storage.id

  container_registry_id = azurerm_container_registry.acr.id

  identity {

    type = "SystemAssigned"
  }
}