import warnings

from pydantic.warnings import PydanticDeprecatedSince211

warnings.filterwarnings("ignore", category=PydanticDeprecatedSince211)
warnings.filterwarnings("ignore", message=".*Hub is deprecated.*")
warnings.filterwarnings("ignore", message="The 'warn' method is deprecated, use 'warning' instead")
warnings.filterwarnings("ignore", message=".*UnsupportedFieldAttributeWarning.*")
warnings.filterwarnings("ignore", module="pydantic._internal._generate_schema")
warnings.filterwarnings("ignore", module="weave.telemetry.trace_sentry")
