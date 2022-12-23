#for logging and dynaconf
import logging
from dynaconf import Dynaconf

logging.basicConfig(level=logging.INFO)

settings = Dynaconf(settings_file="settings.toml")