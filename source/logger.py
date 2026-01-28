import structlog
import logging
import sys
from pathlib import Path

def configure_logging(log_file: str = "logs/app.log"):
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(structlog.stdlib.ProcessorFormatter(processor=structlog.dev.ConsoleRenderer()))
    json_handler = logging.FileHandler(log_file)
    json_handler.setFormatter(structlog.stdlib.ProcessorFormatter(processor=structlog.processors.JSONRenderer()))
    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            console_handler,
            json_handler,
        ]
    )
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    logging.getLogger("giskard.llm.embeddings").setLevel(logging.ERROR)
    logging.getLogger("ragas.llms.base").setLevel(logging.ERROR)

def get_logger(name: str = None):
    return structlog.get_logger(name)
