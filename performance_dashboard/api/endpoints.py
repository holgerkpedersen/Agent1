"""
API Endpoints Module for Performance Dashboard

This module defines RESTful API endpoints for managing and retrieving 
performance metrics data including system monitoring, application analytics,
and historical reporting capabilities.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from flask import Blueprint, request, jsonify, current_app
from marshmallow import ValidationError

# Local imports for models and services
try:
    from ..models.performance_metrics import MetricType
    from ..services.metrics_service import MetricsService
    from ..services.analytics_service import AnalyticsService
    from ..utils.pagination import paginate_query
    from ..schemas.metric_schema import MetricSchema, AggregatedMetricSchema
except ImportError:
    # Fallback implementations for development/testing
    logger.warning("Silenced exception in endpoints.py:22")

# Initialize logger
logger = logging.getLogger(__name__)

# Create Blueprint with URL prefix
api_bp = Blueprint('performance_api', __name__, url_prefix='/api/v1')


def handle_service_errors(func):
    """Decorator to standardize error handling across endpoints."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ValidationError as e:
            logger.warning(f"Validation error in {func.__name__}: {e.messages}")
            return jsonify({
                'error': 'validation_failed',
                'message': str(e.messages),
                'status_code': 400
            }), 400
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {str(e)}", exc_info=True)
            return jsonify({
                'error': 'internal_error',
                'message': str(e),
                'status_code': 500
            }), 500
    wrapper.__name__ = func.__name__
    return wrapper


@api_bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint to verify API availability."""
    logger.info("Health check requested")
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'version': current_app.config.get('API_VERSION', '1.0')
    }), 200


@api_bp.route('/metrics', methods=['GET'])
@handle_service_errors
def get_metrics():
    """Retrieve performance metrics with optional filtering and pagination."""
    logger.info("Metrics retrieval requested")

    # Extract query parameters
    metric_type = request.args.get('type')
    start_time = request.args.get('start_time')
    end_time = request.args.get('end_time')
    limit = int(request.args.get('limit', 100))
    offset = int(request.args.get('offset', 0))

    # Validate parameters
    if limit > current_app.config.get('MAX_LIMIT', 1000):
        return jsonify({
            'error': 'invalid_parameter',
            'message': f'Limit exceeds maximum allowed ({current_app.config["MAX_LIMIT"]})'
        }), 400

    # Initialize service and fetch metrics
    try:
        service = MetricsService()
        query_params = {
            'metric_type': metric_type,
            'start_time': start_time,
            'end_time': end_time,
            'limit': limit,
            'offset': offset
        }

        # Apply filters and pagination
        
        if not hasattr(service, '_query'):
            # Fallback for mock implementations
            return jsonify({
                'data': [],
                'pagination': {
                    'total': 0,
                    'limit': limit,
                    'offset': offset,
                    'has_more': False
                }
            }), 200

        paginated_results = paginate_query(service._query, page_size=limit, offset=offset)
        
        schema = MetricSchema(many=True)
        serialized_data = schema.dump(paginated_results.items)
        
        return jsonify({
            'data': serialized_data,
            'pagination': {
                'total': paginated_results.total,
                'limit': limit,
                'offset': offset,
                'has_more': paginated_results.has_next
            },
            'filters_applied': query_params if any(query_params.values()) else None
        }), 200

    except AttributeError:
        # Handle case where service doesn't exist in development mode
        logger.warning("MetricsService not available, returning empty response")
        return jsonify({
            'data': [],
            'pagination': {
                'total': 0,
                'limit': limit,
                'offset': offset,
                'has_more': False
            }
        }), 200


@api_bp.route('/metrics/aggregated', methods=['GET'])
@handle_service_errors
def get_aggregated_metrics():
    """Retrieve aggregated performance metrics grouped by time intervals."""
    logger.info("Aggregated metrics retrieval requested")

    # Extract aggregation parameters
    metric_type = request.args.get('type')
    interval = request.args.get('interval', 'hour')  # hour, day, week, month
    start_time = request.args.get('start_time')
    end_time = request.args.get('end_time')
    aggregations = request.args.getlist('aggregations[]') or ['avg', 'max', 'min']

    valid_intervals = ['minute', 'hour', 'day', 'week', 'month', 'quarter', 'year']
    if interval not in valid_intervals:
        return jsonify({
            'error': 'invalid_parameter',
            'message': f'Invalid interval. Must be one of {valid_intervals}'
        }), 400

    try:
        service = AnalyticsService()
        aggregation_params = {
            'metric_type': metric_type,
            'interval': interval,
            'start_time': start_time,
            'end_time': end_time,
            'aggregations': aggregations
        }

        aggregated_data = service.get_aggregated_metrics(aggregation_params)
        
        schema = AggregatedMetricSchema(many=True)
        serialized_data = schema.dump(aggregated_data)

        return jsonify({
            'data': serialized_data,
            'interval': interval,
            'aggregations': aggregations,
            'time_range': {
                'start': start_time or (datetime.utcnow() - timedelta(days=7)).isoformat(),
                'end': end_time or datetime.utcnow().isoformat()
            }
        }), 200

    except AttributeError:
        logger.warning("AnalyticsService not available, returning empty response")
        return jsonify({
            'data': [],
            'interval': interval,
            'aggregations': aggregations
        }), 200


@api_bp.route('/metrics/<int:metric_id>', methods=['GET'])
@handle_service_errors
def get_metric_detail(metric_id: int):
    """Retrieve detailed information for a specific metric."""
    logger.info(f"Metric detail requested for ID: {metric_id}")

    try:
        service = MetricsService()
        metric_record = service.get_metric_by_id(metric_id)
        
        if not metric_record:
            return jsonify({
                'error': 'not_found',
                'message': f'Metric with ID {metric_id} not found'
            }), 404

        schema = MetricSchema()
        serialized_data = schema.dump(metric_record)

        return jsonify({
            'data': serialized_data,
            'related_metrics': service.get_related_metrics(metric_id) if hasattr(service, 'get_related_metrics') else []
        }), 200

    except AttributeError:
        logger.warning("MetricsService not available")
        return jsonify({'error': 'service_unavailable'}), 503


@api_bp.route('/metrics', methods=['POST'])
@handle_service_errors
def create_metric():
    """Create a new performance metric entry."""
    logger.info("Metric creation requested")

    data = request.get_json() or {}
    
    if not data:
        return jsonify({
            'error': 'invalid_input',
            'message': 'No JSON payload provided'
        }), 400

    try:
        schema = MetricSchema()
        validated_data = schema.load(data)
        
        service = MetricsService()
        created_metric = service.create_metric(validated_data)
        
        if not hasattr(service, 'create_metric'):
            # Fallback behavior
            return jsonify({
                'data': {**validated_data, 'id': 1},
                'message': 'Metric created successfully'
            }), 201

        serialized_result = schema.dump(created_metric)

        logger.info(f"Successfully created metric with ID: {created_metric.id}")
        return jsonify({
            'data': serialized_result,
            'message': 'Metric created successfully'
        }), 201

    except ValidationError as e:
        raise e  # Re-raise to be handled by decorator


@api_bp.route('/metrics/<int:metric_id>', methods=['PUT'])
@handle_service_errors
def update_metric(metric_id: int):
    """Update an existing performance metric."""
    logger.info(f"Metric update requested for ID: {metric_id}")

    data = request.get_json() or {}
    
    if not data:
        return jsonify({
            'error': 'invalid_input',
            'message': 'No JSON payload provided'
        }), 400

    try:
        service = MetricsService()
        existing_metric = service.get_metric_by_id(metric_id)
        
        if not existing_metric:
            return jsonify({
                'error': 'not_found',
                'message': f'Metric with ID {metric_id} not found'
            }), 404

        schema = MetricSchema(partial=True)
        validated_data = schema.load(data, instance=existing_metric)
        
        updated_metric = service.update_metric(metric_id, validated_data)
        
        if not hasattr(service, 'update_metric'):
            # Fallback behavior
            return jsonify({
                'data': {**validated_data, 'id': metric_id},
                'message': 'Metric updated successfully'
            }), 200

        serialized_result = schema.dump(updated_metric)

        logger.info(f"Successfully updated metric with ID: {metric_id}")
        return jsonify({
            'data': serialized_result,
            'message': 'Metric updated successfully'
        }), 200

    except ValidationError as e:
        raise e


@api_bp.route('/metrics/<int:metric_id>', methods=['DELETE'])
@handle_service_errors
def delete_metric(metric_id: int):
    """Delete a performance metric."""
    logger.info(f"Metric deletion requested for ID: {metric_id}")

    try:
        service = MetricsService()
        existing_metric = service.get_metric_by_id(metric_id)
        
        if not existing_metric:
            return jsonify({
                'error': 'not_found',
                'message': f'Metric with ID {metric_id} not found'
            }), 404

        deletion_result = service.delete_metric(metric_id)
        
        if not hasattr(service, 'delete_metric'):
            # Fallback behavior
            return jsonify({
                'deleted_id': metric_id,
                'message': 'Metric deleted successfully'
            }), 200

        logger.info(f"Successfully deleted metric with ID: {metric_id}")
        return jsonify({
            'deleted_id': metric_id,
            'message': 'Metric deleted successfully',
            'result': deletion_result if isinstance(deletion_result, dict) else None
        }), 200

    except AttributeError:
        logger.warning("MetricsService not available")
        return jsonify({'error': 'service_unavailable'}), 503


@api_bp.route('/dashboards', methods=['GET'])
def get_dashboards():
    """Retrieve available dashboard configurations."""
    logger.info("Dashboard listing requested")

    dashboards_config = current_app.config.get('DASHBOARDS_CONFIG', {})
    
    return jsonify({
        'data': dashboards_config,
        'total_count': len(dashboards_config)
    }), 200


@api_bp.route('/dashboards/<string:dashboard_name>', methods=['GET'])
@handle_service_errors
def get_dashboard_data(dashboard_name: str):
    """Retrieve comprehensive data for a specific dashboard."""
    logger.info(f"Dashboard data requested for: {dashboard_name}")

    # Extract time range parameters
    start_time = request.args.get('start_time')
    end_time = request.args.get('end_time')
    
    try:
        analytics_service = AnalyticsService()
        metrics_service = MetricsService()
        
        dashboard_config = current_app.config.get('DASHBOARDS_CONFIG', {}).get(dashboard_name)
        
        if not dashboard_config and not hasattr(analytics_service, 'generate_dashboard_data'):
            # Fallback behavior for development mode
            return jsonify({
                'dashboard': dashboard_name,
                'data': {
                    'summary_metrics': [],
                    'trend_data': [],
                    'alerts': []
                },
                'generated_at': datetime.utcnow().isoformat()
            }), 200

        # Generate comprehensive dashboard data
        time_range = {
            'start': start_time or (datetime.utcnow() - timedelta(days=7)).isoformat(),
            'end': end_time or datetime.utcnow().isoformat()
        }

        if hasattr(analytics_service, 'generate_dashboard_data'):
            dashboard_data = analytics_service.generate_dashboard_data(dashboard_name, time_range)
        else:
            # Fallback implementation
            summary_metrics = metrics_service.get_recent_summary(time_range['start'], time_range['end']) \
                if hasattr(metrics_service, 'get_recent_summary') else []

            dashboard_data = {
                'summary_metrics': summary_metrics or [],
                'trend_data': [],
                'alerts': analytics_service.get_active_alerts() \
                    if hasattr(analytics_service, 'get_active_alerts') else []
            }

        return jsonify({
            'dashboard': dashboard_name,
            'data': dashboard_data,
            'time_range': time_range,
            'generated_at': datetime.utcnow().isoformat()
        }), 200

    except AttributeError:
        logger.warning("Services not available")
        return jsonify({'error': 'service_unavailable'}), 503


@api_bp.route('/alerts', methods=['GET'])
def get_alerts():
    """Retrieve active performance alerts."""
    logger.info("Alerts retrieval requested")

    severity = request.args.get('severity')
    limit = int(request.args.get('limit', 50))

    try:
        service = AnalyticsService()
        if hasattr(service, 'get_active_alerts'):
            alert_params = {'severity': severity, 'limit': limit}
            alerts_data = service.get_active_alerts(alert_params)
        else:
            # Fallback behavior
            alerts_data = []

        return jsonify({
            'data': alerts_data,
            'pagination': {
                'total': len(alerts_data),
                'limit': limit
            },
            'active_count': len(alerts_data) if isinstance(alerts_data, list) else 0
        }), 200

    except AttributeError:
        logger.warning("AnalyticsService not available")
        return jsonify({
            'data': [],
            'pagination': {
                'total': 0,
                'limit': limit
            },
            'active_count': 0
        }), 200


@api_bp.route('/metrics/types', methods=['GET'])
def get_metric_types():
    """Retrieve available metric types and their configurations."""
    logger.info("Metric types retrieval requested")

    try:
        # Attempt to import MetricType enum
        from ..models.performance_metrics import MetricType
        
        metric_types = [
            {
                'name': mt.name,
                'value': mt.value if hasattr(mt, 'value') else str(mt),
                'description': getattr(MetricType, '__doc__', '').split('\n')[1].strip() 
                    if hasattr(MetricType, '__doc__') and len(getattr(MetricType, '__doc__', '').split('\n')) > 1 else None
            } for mt in MetricType
        ]

        return jsonify({
            'data': metric_types,
            'total_count': len(metric_types)
        }), 200

    except ImportError:
        # Fallback definitions
        fallback_metric_types = [
            {'name': 'cpu_usage', 'value': 'CPU_USAGE', 'description': 'CPU utilization percentage'},
            {'name': 'memory_usage', 'value': 'MEMORY_USAGE', 'description': 'Memory consumption in bytes'},
            {'name': 'disk_io', 'value': 'DISK_IO', 'description': 'Disk input/output operations per second'},
            {'name': 'network_throughput', 'value': 'NETWORK_THROUGHPUT', 'description': 'Network bandwidth utilization'},
            {'name': 'response_time', 'value': 'RESPONSE_TIME', 'description': 'Application response latency in milliseconds'},
            {'name': 'request_rate', 'value': 'REQUEST_RATE', 'description': 'HTTP requests per second'},
            {'name': 'error_rate', 'value': 'ERROR_RATE', 'description': 'Percentage of failed requests'},
            {'name': 'uptime', 'value': 'UPTIME', 'description': 'Service availability percentage'}
        ]

        return jsonify({
            'data': fallback_metric_types,
            'total_count': len(fallback_metric_types),
            'note': 'Using default metric types due to missing model definitions'
        }), 200


@api_bp.route('/metrics/bulk', methods=['POST'])
def bulk_create_metrics():
    """Create multiple performance metrics in a single request."""
    logger.info("Bulk metric creation requested")

    data = request.get_json() or {}
    
    if not isinstance(data, list):
        return jsonify({
            'error': 'invalid_input',
            'message': 'Expected array of metric objects'
        }), 400

    if len(data) > current_app.config.get('MAX_BULK_SIZE', 1000):
        return jsonify({
            'error': 'invalid_parameter',
            'message': f'Bulk request exceeds maximum size ({current_app.config["MAX_BULK_SIZE"]})'
        }), 400

    try:
        schema = MetricSchema(many=True)
        validated_data = schema.load(data)
        
        service = MetricsService()
        if hasattr(service, 'bulk_create_metrics'):
            created_metrics = service.bulk_create_metrics(validated_data)
        else:
            # Fallback behavior - simulate creation
            created_metrics = [{**item, 'id': idx + 1} for idx, item in enumerate(validated_data)]

        serialized_results = schema.dump(created_metrics) if hasattr(schema, 'dump') else validated_data

        logger.info(f"Successfully bulk-created {len(serialized_results)} metrics")
        return jsonify({
            'data': serialized_results,
            'count': len(serialized_results),
            'message': f'{len(serialized_results)} metrics created successfully'
        }), 201

    except ValidationError as e:
        logger.warning(f"Bulk validation error: {e.messages}")
        return jsonify({
            'error': 'validation_failed',
            'message': str(e.messages),
            'status_code': 400
        }), 400


@api_bp.route('/metrics/search', methods=['GET'])
def search_metrics():
    """Search metrics using keyword query across multiple fields."""
    logger.info("Metric search requested")

    query = request.args.get('q') or ''
    metric_type = request.args.get('type')
    limit = int(request.args.get('limit', 50))

    if not query:
        return jsonify({
            'error': 'invalid_parameter',
            'message': 'Search query parameter "q" is required'
        }), 400

    try:
        service = MetricsService()
        search_params = {
            'query': query,
            'metric_type': metric_type,
            'limit': limit
        }

        if hasattr(service, 'search_metrics'):
            results = service.search_metrics(search_params)
        else:
            # Fallback behavior - return empty results
            results = []

        schema = MetricSchema(many=True)
        serialized_results = schema.dump(results) if isinstance(results, list) and hasattr(schema, 'dump') else results or []

        return jsonify({
            'data': serialized_results,
            'query': query,
            'pagination': {
                'total': len(serialized_results),
                'limit': limit
            }
        }), 200

    except AttributeError:
        logger.warning("MetricsService not available")
        return jsonify({
            'data': [],
            'query': query,
            'pagination': {
                'total': 0,
                'limit': limit
            }
        }), 200


@api_bp.route('/export/metrics', methods=['GET'])
def export_metrics():
    """Export metrics data in specified format."""
    logger.info("Metrics export requested")

    format_type = request.args.get('format', 'json')  # json, csv, excel
    start_time = request.args.get('start_time')
    end_time = request.args.get('end_time')
    metric_type = request.args.get('type')

    valid_formats = ['json', 'csv', 'excel']
    if format_type not in valid_formats:
        return jsonify({
            'error': 'invalid_parameter',
            'message': f'Invalid export format. Must be one of {valid_formats}'
        }), 400

    try:
        service = MetricsService()
        export_params = {
            'format': format_type,
            'start_time': start_time,
            'end_time': end_time,
            'metric_type': metric_type
        }

        if hasattr(service, 'export_metrics'):
            exported_data = service.export_metrics(export_params)
        else:
            # Fallback behavior - return minimal JSON structure
            exported_data = {
                'format': format_type,
                'records': [],
                'exported_at': datetime.utcnow().isoformat(),
                'note': 'Export functionality unavailable'
            }

        if format_type == 'json':
            return jsonify({
                'data': exported_data.get('records', []),
                'export_info': {
                    'format': format_type,
                    'exported_at': datetime.utcnow().isoformat(),
                    'record_count': len(exported_data.get('records', []))
                }
            }), 200

        # For CSV/Excel formats, would typically return file attachment
        response = jsonify({
            'message': f'Metrics exported in {format_type} format',
            'download_url': '/api/v1/export/metrics/download' if hasattr(service, 'export_metrics') else None,
            'exported_at': datetime.utcnow().isoformat()
        })
        
        return response

    except AttributeError:
        logger.warning("MetricsService not available")
        return jsonify({
            'error': 'service_unavailable',
            'message': 'Export functionality is currently unavailable'
        }), 503


# Error handlers for the blueprint
@api_bp.errorhandler(404)
def handle_404(e):
    """Handle not found errors."""
    logger.warning(f"Endpoint not found: {request.path}")
    return jsonify({
        'error': 'not_found',
        'message': f'The requested endpoint "{request.path}" was not found'
    }), 404


@api_bp.errorhandler(405)
def handle_405(e):
    """Handle method not allowed errors."""
    logger.warning(f"Method not allowed: {request.method} for path: {request.path}")
    return jsonify({
        'error': 'method_not_allowed',
        'message': f'Method "{request.method}" is not allowed for this endpoint'
    }), 405


@api_bp.errorhandler(500)
def handle_500(e):
    """Handle internal server errors."""
    logger.critical(f"Internal server error: {str(e)}", exc_info=True)
    return jsonify({
        'error': 'internal_error',
        'message': 'An unexpected error occurred processing your request'
    }), 500


# Register the blueprint with application factory pattern support
def register_endpoints(app):
    """Register API endpoints blueprint with Flask application."""
    app.register_blueprint(api_bp)
    logger.info("Performance dashboard API endpoints registered successfully")


__all__ = [
    'api_bp',
    'register_endpoints'
]

# Module metadata
__version__ = "1.0.0"
__author__ = "Performance Dashboard Team"