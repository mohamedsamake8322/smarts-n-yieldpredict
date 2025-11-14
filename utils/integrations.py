import os
import logging
import requests
import json
from typing import List, Dict, Optional
from datetime import datetime
import tempfile

logger = logging.getLogger(__name__)

class ExternalIntegrations:
    """Handle integrations with external services like Google Drive, Dropbox, etc."""
    
    def __init__(self):
        self.supported_services = ['googledrive', 'dropbox', 'onedrive']
        self.api_keys = {
            'google_client_id': os.environ.get('GOOGLE_CLIENT_ID'),
            'google_client_secret': os.environ.get('GOOGLE_CLIENT_SECRET'),
            'dropbox_api_key': os.environ.get('DROPBOX_API_KEY'),
            'onedrive_client_id': os.environ.get('ONEDRIVE_CLIENT_ID')
        }

    def get_authorization_url(self, service: str, redirect_uri: str) -> Dict:
        """Get OAuth authorization URL for external service"""
        try:
            if service == 'googledrive':
                return self._get_google_auth_url(redirect_uri)
            elif service == 'dropbox':
                return self._get_dropbox_auth_url(redirect_uri)
            elif service == 'onedrive':
                return self._get_onedrive_auth_url(redirect_uri)
            else:
                return {'error': f'Service non supporté: {service}'}
                
        except Exception as e:
            logger.error(f"Error getting auth URL for {service}: {str(e)}")
            return {'error': f'Erreur d\'autorisation: {str(e)}'}

    def _get_google_auth_url(self, redirect_uri: str) -> Dict:
        """Get Google Drive authorization URL"""
        if not self.api_keys['google_client_id']:
            return {'error': 'Google Client ID non configuré'}
        
        base_url = "https://accounts.google.com/o/oauth2/auth"
        params = {
            'client_id': self.api_keys['google_client_id'],
            'redirect_uri': redirect_uri,
            'scope': 'https://www.googleapis.com/auth/drive.readonly',
            'response_type': 'code',
            'access_type': 'offline'
        }
        
        auth_url = base_url + '?' + '&'.join([f"{k}={v}" for k, v in params.items()])
        
        return {
            'auth_url': auth_url,
            'service': 'googledrive',
            'scope': 'drive.readonly'
        }

    def list_files(self, service: str, access_token: str, folder_id: str = None) -> Dict:
        """List files from external service"""
        try:
            if service == 'googledrive':
                return self._list_google_drive_files(access_token, folder_id)
            elif service == 'dropbox':
                return self._list_dropbox_files(access_token, folder_id)
            else:
                return {'error': f'Service non supporté: {service}'}
                
        except Exception as e:
            logger.error(f"Error listing files from {service}: {str(e)}")
            return {'error': f'Erreur de listage: {str(e)}'}

    def _list_google_drive_files(self, access_token: str, folder_id: str = None) -> Dict:
        """List files from Google Drive"""
        try:
            headers = {'Authorization': f'Bearer {access_token}'}
            
            # Build query
            query = "mimeType != 'application/vnd.google-apps.folder'"
            if folder_id:
                query += f" and '{folder_id}' in parents"
            
            url = f"https://www.googleapis.com/drive/v3/files"
            params = {
                'q': query,
                'fields': 'files(id,name,mimeType,size,createdTime,modifiedTime)',
                'pageSize': 100
            }
            
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            
            data = response.json()
            files = data.get('files', [])
            
            # Filter supported file types
            supported_mimes = {
                'application/pdf': 'pdf',
                'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': 'xlsx',
                'application/vnd.ms-excel': 'xls',
                'text/csv': 'csv',
                'text/plain': 'txt',
                'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 'docx',
                'application/vnd.openxmlformats-officedocument.presentationml.presentation': 'pptx',
                'image/jpeg': 'jpg',
                'image/png': 'png'
            }
            
            filtered_files = []
            for file in files:
                if file['mimeType'] in supported_mimes:
                    filtered_files.append({
                        'id': file['id'],
                        'name': file['name'],
                        'type': supported_mimes[file['mimeType']],
                        'size': int(file.get('size', 0)),
                        'created': file.get('createdTime'),
                        'modified': file.get('modifiedTime'),
                        'service': 'googledrive'
                    })
            
            return {
                'files': filtered_files,
                'total': len(filtered_files),
                'service': 'googledrive'
            }
            
        except requests.RequestException as e:
            return {'error': f'Erreur Google Drive API: {str(e)}'}

    def download_file(self, service: str, file_id: str, access_token: str) -> Dict:
        """Download file from external service"""
        try:
            if service == 'googledrive':
                return self._download_google_drive_file(file_id, access_token)
            elif service == 'dropbox':
                return self._download_dropbox_file(file_id, access_token)
            else:
                return {'error': f'Service non supporté: {service}'}
                
        except Exception as e:
            logger.error(f"Error downloading file from {service}: {str(e)}")
            return {'error': f'Erreur de téléchargement: {str(e)}'}

    def _download_google_drive_file(self, file_id: str, access_token: str) -> Dict:
        """Download file from Google Drive"""
        try:
            headers = {'Authorization': f'Bearer {access_token}'}
            
            # Get file metadata first
            metadata_url = f"https://www.googleapis.com/drive/v3/files/{file_id}"
            metadata_params = {'fields': 'name,mimeType,size'}
            
            metadata_response = requests.get(metadata_url, headers=headers, params=metadata_params)
            metadata_response.raise_for_status()
            metadata = metadata_response.json()
            
            # Download file content
            download_url = f"https://www.googleapis.com/drive/v3/files/{file_id}"
            download_params = {'alt': 'media'}
            
            download_response = requests.get(download_url, headers=headers, params=download_params)
            download_response.raise_for_status()
            
            # Save to temporary file
            temp_file = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=f".{metadata['name'].split('.')[-1] if '.' in metadata['name'] else 'txt'}"
            )
            
            temp_file.write(download_response.content)
            temp_file.close()
            
            return {
                'file_path': temp_file.name,
                'original_name': metadata['name'],
                'mime_type': metadata['mimeType'],
                'size': int(metadata.get('size', 0)),
                'service': 'googledrive'
            }
            
        except requests.RequestException as e:
            return {'error': f'Erreur téléchargement Google Drive: {str(e)}'}

class WebhookManager:
    """Manage webhooks for external notifications"""
    
    def __init__(self):
        self.registered_webhooks = {}

    def register_webhook(self, event_type: str, url: str, secret: str = None) -> str:
        """Register a webhook for specific events"""
        webhook_id = str(uuid.uuid4())
        
        self.registered_webhooks[webhook_id] = {
            'event_type': event_type,
            'url': url,
            'secret': secret,
            'created_at': datetime.now(timezone.utc),
            'active': True
        }
        
        return webhook_id

    def trigger_webhook(self, event_type: str, data: Dict):
        """Trigger webhooks for specific event"""
        try:
            for webhook_id, webhook in self.registered_webhooks.items():
                if webhook['event_type'] == event_type and webhook['active']:
                    self._send_webhook(webhook, data)
                    
        except Exception as e:
            logger.error(f"Error triggering webhooks: {str(e)}")

    def _send_webhook(self, webhook: Dict, data: Dict):
        """Send webhook notification"""
        try:
            payload = {
                'event_type': webhook['event_type'],
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'data': data
            }
            
            headers = {'Content-Type': 'application/json'}
            
            if webhook.get('secret'):
                import hmac
                import hashlib
                signature = hmac.new(
                    webhook['secret'].encode(),
                    json.dumps(payload).encode(),
                    hashlib.sha256
                ).hexdigest()
                headers['X-Webhook-Signature'] = f'sha256={signature}'
            
            response = requests.post(
                webhook['url'],
                json=payload,
                headers=headers,
                timeout=10
            )
            
            logger.info(f"Webhook sent to {webhook['url']}, status: {response.status_code}")
            
        except Exception as e:
            logger.error(f"Failed to send webhook: {str(e)}")

class PublicAPIManager:
    """Manage public REST API for external integrations"""
    
    def __init__(self):
        self.api_keys = {}  # In production, store in database

    def generate_api_key(self, user_id: int, name: str = None) -> str:
        """Generate API key for user"""
        import secrets
        api_key = f"va_{secrets.token_urlsafe(32)}"
        
        self.api_keys[api_key] = {
            'user_id': user_id,
            'name': name,
            'created_at': datetime.now(timezone.utc),
            'active': True,
            'usage_count': 0
        }
        
        return api_key

    def validate_api_key(self, api_key: str) -> Optional[int]:
        """Validate API key and return user_id"""
        if api_key in self.api_keys and self.api_keys[api_key]['active']:
            self.api_keys[api_key]['usage_count'] += 1
            return self.api_keys[api_key]['user_id']
        return None

    def revoke_api_key(self, api_key: str) -> bool:
        """Revoke an API key"""
        if api_key in self.api_keys:
            self.api_keys[api_key]['active'] = False
            return True
        return False