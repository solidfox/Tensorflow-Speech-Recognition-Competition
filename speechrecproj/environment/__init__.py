from config import EnvironmentConfig
import pkgutil


__author__ = 'Daniel Schlaug'

running_on_hops = pkgutil.find_loader('hops') is not None

