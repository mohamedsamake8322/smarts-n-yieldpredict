from functools import wraps
from flask import jsonify, request, session
from flask_login import current_user
import logging

logger = logging.getLogger(__name__)

def login_required_api(f):
    """Decorator for API endpoints that require authentication"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated:
            return jsonify({'error': 'Authentication required'}), 401
        return f(*args, **kwargs)
    return decorated_function

def admin_required(f):
    """Decorator for endpoints that require admin privileges"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated:
            return jsonify({'error': 'Authentication required'}), 401
        if not current_user.is_admin:
            return jsonify({'error': 'Admin privileges required'}), 403
        return f(*args, **kwargs)
    return decorated_function

def check_document_permission(document, permission='read'):
    """Check if current user has permission to access document"""
    if not current_user.is_authenticated:
        return False
    
    # Owner has all permissions
    if document.user_id == current_user.id:
        return True
    
    # Check shared permissions
    from models import DocumentShare
    share = DocumentShare.query.filter_by(
        document_id=document.id,
        shared_with_user_id=current_user.id
    ).first()
    
    if share:
        if permission == 'read' and share.permission in ['read', 'write', 'admin']:
            return True
        elif permission == 'write' and share.permission in ['write', 'admin']:
            return True
        elif permission == 'admin' and share.permission == 'admin':
            return True
    
    return False

def check_workspace_permission(workspace, permission='read'):
    """Check if current user has permission to access workspace"""
    if not current_user.is_authenticated:
        return False
    
    # Owner has all permissions
    if workspace.owner_id == current_user.id:
        return True
    
    # Check membership permissions
    from models import WorkspaceMember
    member = WorkspaceMember.query.filter_by(
        workspace_id=workspace.id,
        user_id=current_user.id
    ).first()
    
    if member:
        if permission == 'read' and member.role in ['viewer', 'editor', 'admin']:
            return True
        elif permission == 'write' and member.role in ['editor', 'admin']:
            return True
        elif permission == 'admin' and member.role == 'admin':
            return True
    
    return False

def log_analytics_event(event_type, event_data=None):
    """Log analytics event"""
    try:
        from models import Analytics, db
        
        analytics = Analytics(
            user_id=current_user.id if current_user.is_authenticated else None,
            event_type=event_type,
            event_data=event_data,
            session_id=session.get('session_id'),
            ip_address=request.remote_addr,
            user_agent=request.headers.get('User-Agent', '')[:500]
        )
        
        db.session.add(analytics)
        db.session.commit()
        
    except Exception as e:
        logger.error(f"Failed to log analytics event: {e}")

def get_user_language():
    """Get user's preferred language"""
    if current_user.is_authenticated:
        return current_user.preferred_language
    return session.get('language', 'fr')

def set_user_language(language):
    """Set user's preferred language"""
    if current_user.is_authenticated:
        current_user.preferred_language = language
        from models import db
        db.session.commit()
    session['language'] = language