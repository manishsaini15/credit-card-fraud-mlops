# =============================================================================
# Configuration Utility
# =============================================================================
# This module loads configuration values from config/schema.yaml.
# All project components should use this instead of hardcoding values.
# =============================================================================

# Import YAML library
import yaml


# Function to load schema configuration
def load_schema():

    # Open the schema file
    with open("config/schema.yaml", "r") as file:

        # Read YAML and convert to Python dictionary
        schema = yaml.safe_load(file)

    # Return configuration
    return schema