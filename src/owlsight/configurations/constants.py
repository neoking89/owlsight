from owlsight.configurations.schema import Schema


CONFIG_DEFAULTS = Schema.get_config_defaults()
CONFIG_CHOICES = Schema.get_config_choices()
CONFIG_DESCRIPTIONS = Schema.get_config_descriptions()
CONFIG_TYPES = Schema.get_config_types()
MAIN_MENU = Schema.get_main_menu()
ASSISTENT_PROMPT = list(MAIN_MENU.keys())[0]
