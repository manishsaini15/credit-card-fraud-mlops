output "resource_group_name" {

  value = azurerm_resource_group.rg.name
}

output "machine_learning_workspace" {

  value = azurerm_machine_learning_workspace.ml_workspace.name
}

output "storage_account_name" {

  value = azurerm_storage_account.storage.name
}

output "container_registry_name" {

  value = azurerm_container_registry.acr.name
}