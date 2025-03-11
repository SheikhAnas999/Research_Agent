"""
Advanced logging utility for AI Research Agent
Provides structured logging with real-time monitoring
"""

import logging
import logging.handlers
import os
import sys
import json
import traceback
from datetime import datetime
from typing import Dict, Any, Optional
import colorlog
from functools import wraps
import time


class ResearchAgentLogger:
    """Custom logger for research agent with structured logging"""
    
    def __init__(self, name: str, log_file: str = "logs/research_agent.log", level: str = "INFO"):
        self.name = name
        self.log_file = log_file
        self.level = level
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        """Setup logger with file and console handlers"""
        
        # Create logs directory
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        
        # Create logger
        logger = logging.getLogger(self.name)
        logger.setLevel(getattr(logging, self.level.upper(), logging.INFO))
        
        # Remove existing handlers to avoid duplicates
        logger.handlers.clear()
        
        # File handler with rotation
        file_handler = logging.handlers.RotatingFileHandler(
            self.log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        
        # Console handler with colors
        console_handler = colorlog.StreamHandler(sys.stdout)
        
        # Formatters
        file_formatter = logging.Formatter(
            '%(asctime)s | %(name)s | %(levelname)s | %(funcName)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        console_formatter = colorlog.ColoredFormatter(
            '%(log_color)s%(asctime)s | %(name)s | %(levelname)s | %(funcName)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red,bg_white',
            }
        )
        
        # Set formatters
        file_handler.setFormatter(file_formatter)
        console_handler.setFormatter(console_formatter)
        
        # Add handlers
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def log_structured(self, level: str, message: str, extra_data: Dict[str, Any] = None):
        """Log structured data with JSON formatting"""
        structured_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "logger": self.name,
            "level": level.upper(),
            "message": message,
            "extra": extra_data or {}
        }
        
        log_method = getattr(self.logger, level.lower(), self.logger.info)
        log_method(f"STRUCTURED: {json.dumps(structured_data, default=str)}")
    
    def log_api_call(self, api_name: str, endpoint: str, method: str = "GET", 
                     status_code: Optional[int] = None, response_time: Optional[float] = None,
                     error: Optional[str] = None):
        """Log API calls with structured data"""
        api_data = {
            "api_name": api_name,
            "endpoint": endpoint,
            "method": method,
            "status_code": status_code,
            "response_time": response_time,
            "error": error,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if error:
            self.log_structured("error", f"API call failed: {api_name}", api_data)
        else:
            self.log_structured("info", f"API call successful: {api_name}", api_data)
    
    def log_paper_processing(self, paper_title: str, action: str, success: bool, 
                           details: Dict[str, Any] = None):
        """Log paper processing events"""
        paper_data = {
            "paper_title": paper_title[:100] + "..." if len(paper_title) > 100 else paper_title,
            "action": action,
            "success": success,
            "details": details or {},
            "timestamp": datetime.utcnow().isoformat()
        }
        
        level = "info" if success else "error"
        message = f"Paper {action}: {'SUCCESS' if success else 'FAILED'}"
        self.log_structured(level, message, paper_data)
    
    def log_verification_result(self, paper_id: str, verification_type: str, 
                              result: bool, confidence: float, details: Dict[str, Any] = None):
        """Log verification results"""
        verification_data = {
            "paper_id": paper_id,
            "verification_type": verification_type,
            "result": result,
            "confidence": confidence,
            "details": details or {},
            "timestamp": datetime.utcnow().isoformat()
        }
        
        level = "info" if result else "warning"
        message = f"Verification {verification_type}: {'PASSED' if result else 'FAILED'} (confidence: {confidence:.2f})"
        self.log_structured(level, message, verification_data)
    
    def log_search_results(self, query: str, source: str, results_count: int, 
                          time_taken: float, errors: Optional[List[str]] = None):
        """Log search results"""
        search_data = {
            "query": query,
            "source": source,
            "results_count": results_count,
            "time_taken": time_taken,
            "errors": errors or [],
            "timestamp": datetime.utcnow().isoformat()
        }
        
        level = "info" if not errors else "warning"
        message = f"Search completed: {source} returned {results_count} results in {time_taken:.2f}s"
        self.log_structured(level, message, search_data)
    
    def log_error_with_context(self, error: Exception, context: str, 
                              additional_data: Dict[str, Any] = None):
        """Log errors with full context and traceback"""
        error_data = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context,
            "traceback": traceback.format_exc(),
            "additional_data": additional_data or {},
            "timestamp": datetime.utcnow().isoformat()
        }
        
        self.log_structured("error", f"Error in {context}: {str(error)}", error_data)
    
    def log_performance_metrics(self, operation: str, duration: float, 
                              memory_usage: Optional[float] = None,
                              additional_metrics: Dict[str, Any] = None):
        """Log performance metrics"""
        metrics_data = {
            "operation": operation,
            "duration": duration,
            "memory_usage": memory_usage,
            "additional_metrics": additional_metrics or {},
            "timestamp": datetime.utcnow().isoformat()
        }
        
        self.log_structured("info", f"Performance: {operation} took {duration:.3f}s", metrics_data)
    
    # Standard logging methods
    def debug(self, message: str, extra: Dict[str, Any] = None):
        """Debug level logging"""
        if extra:
            self.log_structured("debug", message, extra)
        else:
            self.logger.debug(message)
    
    def info(self, message: str, extra: Dict[str, Any] = None):
        """Info level logging"""
        if extra:
            self.log_structured("info", message, extra)
        else:
            self.logger.info(message)
    
    def warning(self, message: str, extra: Dict[str, Any] = None):
        """Warning level logging"""
        if extra:
            self.log_structured("warning", message, extra)
        else:
            self.logger.warning(message)
    
    def error(self, message: str, extra: Dict[str, Any] = None):
        """Error level logging"""
        if extra:
            self.log_structured("error", message, extra)
        else:
            self.logger.error(message)
    
    def critical(self, message: str, extra: Dict[str, Any] = None):
        """Critical level logging"""
        if extra:
            self.log_structured("critical", message, extra)
        else:
            self.logger.critical(message)


def setup_logger(name: str, log_file: str = "logs/research_agent.log", level: str = "INFO") -> ResearchAgentLogger:
    """Factory function to create configured logger"""
    return ResearchAgentLogger(name, log_file, level)


def log_execution_time(logger: ResearchAgentLogger):
    """Decorator to log function execution time"""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                logger.log_performance_metrics(func.__name__, duration)
                return result
            except Exception as e:
                duration = time.time() - start_time
                logger.log_error_with_context(e, f"function {func.__name__}", {
                    "duration": duration,
                    "args": str(args)[:200],
                    "kwargs": str(kwargs)[:200]
                })
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                logger.log_performance_metrics(func.__name__, duration)
                return result
            except Exception as e:
                duration = time.time() - start_time
                logger.log_error_with_context(e, f"function {func.__name__}", {
                    "duration": duration,
                    "args": str(args)[:200],
                    "kwargs": str(kwargs)[:200]
                })
                raise
        
        # Return appropriate wrapper based on whether function is async
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def log_api_calls(logger: ResearchAgentLogger, api_name: str):
    """Decorator to log API calls"""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            endpoint = kwargs.get('url', 'unknown')
            method = kwargs.get('method', 'GET')
            
            try:
                result = await func(*args, **kwargs)
                response_time = time.time() - start_time
                
                # Try to extract status code from result
                status_code = None
                if hasattr(result, 'status'):
                    status_code = result.status
                elif hasattr(result, 'status_code'):
                    status_code = result.status_code
                elif isinstance(result, dict) and 'status_code' in result:
                    status_code = result['status_code']
                
                logger.log_api_call(api_name, endpoint, method, status_code, response_time)
                return result
                
            except Exception as e:
                response_time = time.time() - start_time
                logger.log_api_call(api_name, endpoint, method, None, response_time, str(e))
                raise
        
        # Return appropriate wrapper based on whether function is async
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


class LoggingContext:
    """Context manager for structured logging"""
    
    def __init__(self, logger: ResearchAgentLogger, operation: str, **context_data):
        self.logger = logger
        self.operation = operation
        self.context_data = context_data
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        self.logger.info(f"Starting {self.operation}", self.context_data)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        
        if exc_type is None:
            self.logger.log_performance_metrics(self.operation, duration, additional_metrics=self.context_data)
            self.logger.info(f"Completed {self.operation} successfully")
        else:
            self.logger.log_error_with_context(exc_val, self.operation, {
                **self.context_data,
                "duration": duration
            })
        
        return False  # Don't suppress exceptions


# Global logger instance
_global_logger = None

def get_logger(name: str = "research_agent") -> ResearchAgentLogger:
    """Get or create global logger instance"""
    global _global_logger
    if _global_logger is None:
        _global_logger = setup_logger(name)
    return _global_logger:
                    status_code = result['status_code']
                
                logger.log_api_call(api_name, endpoint, method, status_code, response_time)
                return result
                
            except Exception as e:
                response_time = time.time() - start_time
                logger.log_api_call(api_name, endpoint, method, None, response_time, str(e))
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            endpoint = kwargs.get('url', 'unknown')
            method = kwargs.get('method', 'GET')
            
            try:
                result = func(*args, **kwargs)
                response_time = time.time() - start_time
                
                # Try to extract status code from result
                status_code = None
                if hasattr(result, 'status_code'):
                    status_code = result.status_code
                elif isinstance(result, dict) and 'status_code' in result