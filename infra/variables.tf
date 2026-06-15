variable "resource_group_name" {
  description = "Azure Resource Group Name"
  type        = string
}

variable "location" {
  description = "Azure Region"
  type        = string
}

variable "workspace_name" {
  description = "Azure ML Workspace Name"
  type        = string
}

variable "storage_account_name" {
  description = "Storage Account Name"
  type        = string
}

variable "key_vault_name" {
  description = "Azure Key Vault Name"
  type        = string
}

variable "application_insights_name" {
  description = "Application Insights Name"
  type        = string
}

variable "container_registry_name" {
  description = "Azure Container Registry Name"
  type        = string
}