from datetime import datetime, timezone
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
import json

db = SQLAlchemy()

class User(UserMixin, db.Model):
    """User model for authentication and user management"""
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False, index=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(256), nullable=False)
    first_name = db.Column(db.String(64), nullable=False)
    last_name = db.Column(db.String(64), nullable=False)
    preferred_language = db.Column(db.String(5), default='fr')
    active = db.Column(db.Boolean, default=True)
    is_admin = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    last_login = db.Column(db.DateTime)
    
    # Relationships
    documents = db.relationship('Document', backref='owner', lazy='dynamic', cascade='all, delete-orphan')
    conversations = db.relationship('Conversation', backref='user', lazy='dynamic', cascade='all, delete-orphan')
    annotations = db.relationship('Annotation', backref='user', lazy='dynamic', cascade='all, delete-orphan')
    shared_documents = db.relationship('DocumentShare', foreign_keys='DocumentShare.shared_with_user_id', 
                                     backref='shared_user', lazy='dynamic')

    def set_password(self, password):
        """Set password hash"""
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        """Check password against hash"""
        return check_password_hash(self.password_hash, password)

    def get_full_name(self):
        """Get full name"""
        return f"{self.first_name} {self.last_name}"

    def to_dict(self):
        """Convert to dictionary"""
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'full_name': self.get_full_name(),
            'preferred_language': self.preferred_language,
            'is_admin': self.is_admin,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

class Document(db.Model):
    """Document model for storing uploaded files and their content"""
    __tablename__ = 'documents'
    
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    original_filename = db.Column(db.String(255), nullable=False)
    file_type = db.Column(db.String(50), nullable=False)
    file_size = db.Column(db.Integer, nullable=False)
    content_text = db.Column(db.Text, nullable=False)
    detected_language = db.Column(db.String(5), default='en')
    word_count = db.Column(db.Integer, default=0)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc), 
                          onupdate=lambda: datetime.now(timezone.utc))
    is_processed = db.Column(db.Boolean, default=True)
    processing_status = db.Column(db.String(50), default='completed')
    
    # Foreign keys
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, index=True)
    workspace_id = db.Column(db.Integer, db.ForeignKey('workspaces.id'), nullable=True)
    
    # Relationships
    vectors = db.relationship('DocumentVector', backref='document', lazy='dynamic', cascade='all, delete-orphan')
    annotations = db.relationship('Annotation', backref='document', lazy='dynamic', cascade='all, delete-orphan')
    shares = db.relationship('DocumentShare', backref='document', lazy='dynamic', cascade='all, delete-orphan')
    summaries = db.relationship('DocumentSummary', backref='document', lazy='dynamic', cascade='all, delete-orphan')
    versions = db.relationship('DocumentVersion', backref='document', lazy='dynamic', cascade='all, delete-orphan')

    def to_dict(self, include_content=False):
        """Convert to dictionary"""
        result = {
            'id': self.id,
            'filename': self.filename,
            'original_filename': self.original_filename,
            'file_type': self.file_type,
            'file_size': self.file_size,
            'detected_language': self.detected_language,
            'word_count': self.word_count,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'processing_status': self.processing_status,
            'owner': self.owner.get_full_name() if self.owner else None
        }
        if include_content:
            result['content_text'] = self.content_text
        return result

class DocumentVector(db.Model):
    """Vector representation of documents for search"""
    __tablename__ = 'document_vectors'
    
    id = db.Column(db.Integer, primary_key=True)
    document_id = db.Column(db.Integer, db.ForeignKey('documents.id'), nullable=False)
    vector_data = db.Column(db.Text, nullable=False)  # JSON encoded vector
    vector_method = db.Column(db.String(50), default='tfidf')
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    
    def get_vector(self):
        """Get vector as numpy array"""
        import numpy as np
        return np.array(json.loads(self.vector_data))
    
    def set_vector(self, vector):
        """Set vector from numpy array"""
        import numpy as np
        if isinstance(vector, np.ndarray):
            self.vector_data = json.dumps(vector.tolist())
        else:
            self.vector_data = json.dumps(list(vector))

class Workspace(db.Model):
    """Workspace model for organizing documents and collaboration"""
    __tablename__ = 'workspaces'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text)
    is_public = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    
    # Foreign keys
    owner_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    
    # Relationships
    documents = db.relationship('Document', backref='workspace', lazy='dynamic')
    members = db.relationship('WorkspaceMember', backref='workspace', lazy='dynamic', cascade='all, delete-orphan')
    owner = db.relationship('User', backref='owned_workspaces')

    def to_dict(self):
        """Convert to dictionary"""
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'is_public': self.is_public,
            'owner': self.owner.get_full_name(),
            'member_count': self.members.count(),
            'document_count': self.documents.count(),
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

class WorkspaceMember(db.Model):
    """Workspace membership with permissions"""
    __tablename__ = 'workspace_members'
    
    id = db.Column(db.Integer, primary_key=True)
    workspace_id = db.Column(db.Integer, db.ForeignKey('workspaces.id'), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    role = db.Column(db.String(20), default='viewer')  # viewer, editor, admin
    joined_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    
    user = db.relationship('User', backref='workspace_memberships')

class DocumentShare(db.Model):
    """Document sharing between users"""
    __tablename__ = 'document_shares'
    
    id = db.Column(db.Integer, primary_key=True)
    document_id = db.Column(db.Integer, db.ForeignKey('documents.id'), nullable=False)
    shared_by_user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    shared_with_user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    permission = db.Column(db.String(20), default='read')  # read, write, admin
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    expires_at = db.Column(db.DateTime, nullable=True)
    
    shared_by = db.relationship('User', foreign_keys=[shared_by_user_id], backref='shared_documents_by_me')

class Conversation(db.Model):
    """Chat conversation history"""
    __tablename__ = 'conversations'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    title = db.Column(db.String(200))
    language = db.Column(db.String(5), default='fr')
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc), 
                          onupdate=lambda: datetime.now(timezone.utc))
    
    messages = db.relationship('Message', backref='conversation', lazy='dynamic', cascade='all, delete-orphan')

    def to_dict(self):
        """Convert to dictionary"""
        return {
            'id': self.id,
            'title': self.title or f"Conversation du {self.created_at.strftime('%d/%m/%Y')}",
            'language': self.language,
            'message_count': self.messages.count(),
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }

class Message(db.Model):
    """Individual messages in conversations"""
    __tablename__ = 'messages'
    
    id = db.Column(db.Integer, primary_key=True)
    conversation_id = db.Column(db.Integer, db.ForeignKey('conversations.id'), nullable=False)
    role = db.Column(db.String(20), nullable=False)  # user, assistant, system
    content = db.Column(db.Text, nullable=False)
    meta_data = db.Column(db.Text)  # JSON for additional data
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'id': self.id,
            'role': self.role,
            'content': self.content,
            'metadata': json.loads(self.meta_data) if self.meta_data else {},
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

class Annotation(db.Model):
    """Document annotations and comments"""
    __tablename__ = 'annotations'
    
    id = db.Column(db.Integer, primary_key=True)
    document_id = db.Column(db.Integer, db.ForeignKey('documents.id'), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    annotation_type = db.Column(db.String(50), default='comment')  # comment, highlight, note
    content = db.Column(db.Text, nullable=False)
    position_data = db.Column(db.Text)  # JSON for position/selection data
    is_resolved = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'id': self.id,
            'annotation_type': self.annotation_type,
            'content': self.content,
            'position_data': json.loads(self.position_data) if self.position_data else {},
            'is_resolved': self.is_resolved,
            'author': self.user.get_full_name(),
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

class DocumentSummary(db.Model):
    """AI-generated document summaries"""
    __tablename__ = 'document_summaries'
    
    id = db.Column(db.Integer, primary_key=True)
    document_id = db.Column(db.Integer, db.ForeignKey('documents.id'), nullable=False)
    summary_type = db.Column(db.String(50), default='auto')  # auto, manual, key_points
    content = db.Column(db.Text, nullable=False)
    language = db.Column(db.String(5), default='fr')
    confidence_score = db.Column(db.Float, default=0.0)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'id': self.id,
            'summary_type': self.summary_type,
            'content': self.content,
            'language': self.language,
            'confidence_score': self.confidence_score,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

class DocumentVersion(db.Model):
    """Document version history"""
    __tablename__ = 'document_versions'
    
    id = db.Column(db.Integer, primary_key=True)
    document_id = db.Column(db.Integer, db.ForeignKey('documents.id'), nullable=False)
    version_number = db.Column(db.Integer, nullable=False)
    content_text = db.Column(db.Text, nullable=False)
    change_description = db.Column(db.String(500))
    created_by = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    
    creator = db.relationship('User', backref='document_versions_created')

class Analytics(db.Model):
    """Analytics and usage statistics"""
    __tablename__ = 'analytics'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    event_type = db.Column(db.String(50), nullable=False)  # document_upload, search, chat, etc.
    event_data = db.Column(db.Text)  # JSON data
    session_id = db.Column(db.String(100))
    ip_address = db.Column(db.String(45))
    user_agent = db.Column(db.String(500))
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    
    user = db.relationship('User', backref='analytics_events')

class Quiz(db.Model):
    """AI-generated quizzes from documents"""
    __tablename__ = 'quizzes'
    
    id = db.Column(db.Integer, primary_key=True)
    document_id = db.Column(db.Integer, db.ForeignKey('documents.id'), nullable=False)
    title = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text)
    language = db.Column(db.String(5), default='fr')
    difficulty = db.Column(db.String(20), default='medium')  # easy, medium, hard
    created_by = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    
    questions = db.relationship('QuizQuestion', backref='quiz', lazy='dynamic', cascade='all, delete-orphan')
    document = db.relationship('Document', backref='quizzes')
    creator = db.relationship('User', backref='quizzes_created')

class QuizQuestion(db.Model):
    """Individual quiz questions"""
    __tablename__ = 'quiz_questions'
    
    id = db.Column(db.Integer, primary_key=True)
    quiz_id = db.Column(db.Integer, db.ForeignKey('quizzes.id'), nullable=False)
    question_text = db.Column(db.Text, nullable=False)
    question_type = db.Column(db.String(20), default='multiple_choice')
    correct_answer = db.Column(db.Text, nullable=False)
    options = db.Column(db.Text)  # JSON for multiple choice options
    explanation = db.Column(db.Text)
    points = db.Column(db.Integer, default=1)
    order_index = db.Column(db.Integer, default=0)

class EntityExtraction(db.Model):
    """Extracted entities from documents"""
    __tablename__ = 'entity_extractions'
    
    id = db.Column(db.Integer, primary_key=True)
    document_id = db.Column(db.Integer, db.ForeignKey('documents.id'), nullable=False)
    entity_type = db.Column(db.String(50), nullable=False)  # person, organization, date, money, etc.
    entity_value = db.Column(db.String(500), nullable=False)
    confidence_score = db.Column(db.Float, default=0.0)
    context = db.Column(db.Text)  # Surrounding text
    position_start = db.Column(db.Integer)
    position_end = db.Column(db.Integer)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    
    document = db.relationship('Document', backref='extracted_entities')