from typing import Dict, Optional, List
import os
import jwt
import bcrypt
from datetime import datetime, timedelta
import logging
from pathlib import Path
import json
from cryptography.fernet import Fernet
import base64
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

class SecurityManager:
    """Manager for handling security operations."""
    
    def __init__(self, config_path: Optional[Path] = None) -> None:
        """Initialize security manager.
        
        Args:
            config_path: Optional path to security configuration
        """
        load_dotenv()
        
        # Load or generate secret key
        self.secret_key = os.getenv('JWT_SECRET_KEY')
        if not self.secret_key:
            self.secret_key = base64.b64encode(os.urandom(32)).decode('utf-8')
            logger.warning("JWT secret key not found in environment, generated new key")
            
        # Initialize encryption key
        self.encryption_key = os.getenv('ENCRYPTION_KEY')
        if not self.encryption_key:
            self.encryption_key = Fernet.generate_key()
            logger.warning("Encryption key not found in environment, generated new key")
            
        self.fernet = Fernet(self.encryption_key)
        
        # Load configuration if provided
        self.config = self._load_config(config_path) if config_path else {}
        
    def _load_config(self, config_path: Path) -> Dict:
        """Load security configuration.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading security config: {e}")
            return {}
            
    def hash_password(self, password: str) -> bytes:
        """Hash a password using bcrypt.
        
        Args:
            password: Password to hash
            
        Returns:
            Hashed password
        """
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password.encode('utf-8'), salt)
        
    def verify_password(self, password: str, hashed: bytes) -> bool:
        """Verify a password against its hash.
        
        Args:
            password: Password to verify
            hashed: Hashed password to check against
            
        Returns:
            True if password matches hash
        """
        return bcrypt.checkpw(password.encode('utf-8'), hashed)
        
    def generate_token(
        self,
        user_id: str,
        expiration_hours: int = 24,
        additional_claims: Optional[Dict] = None
    ) -> str:
        """Generate a JWT token.
        
        Args:
            user_id: User identifier
            expiration_hours: Token expiration time in hours
            additional_claims: Optional additional claims for token
            
        Returns:
            JWT token string
        """
        expiration = datetime.utcnow() + timedelta(hours=expiration_hours)
        
        claims = {
            'sub': user_id,
            'exp': expiration,
            'iat': datetime.utcnow()
        }
        
        if additional_claims:
            claims.update(additional_claims)
            
        return jwt.encode(claims, self.secret_key, algorithm='HS256')
        
    def verify_token(self, token: str) -> Dict:
        """Verify and decode a JWT token.
        
        Args:
            token: JWT token to verify
            
        Returns:
            Decoded token claims
            
        Raises:
            jwt.InvalidTokenError: If token is invalid
        """
        try:
            return jwt.decode(token, self.secret_key, algorithms=['HS256'])
        except jwt.ExpiredSignatureError:
            logger.error("Token has expired")
            raise
        except jwt.InvalidTokenError as e:
            logger.error(f"Invalid token: {e}")
            raise
            
    def encrypt_data(self, data: str) -> str:
        """Encrypt sensitive data.
        
        Args:
            data: Data to encrypt
            
        Returns:
            Encrypted data string
        """
        return self.fernet.encrypt(data.encode('utf-8')).decode('utf-8')
        
    def decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt encrypted data.
        
        Args:
            encrypted_data: Encrypted data to decrypt
            
        Returns:
            Decrypted data string
        """
        return self.fernet.decrypt(encrypted_data.encode('utf-8')).decode('utf-8')
        
    def mask_sensitive_data(self, data: Dict) -> Dict:
        """Mask sensitive data in dictionary.
        
        Args:
            data: Dictionary containing data to mask
            
        Returns:
            Dictionary with masked sensitive data
        """
        sensitive_fields = [
            'credit_card',
            'ssn',
            'password',
            'account_number',
            'api_key'
        ]
        
        masked = data.copy()
        for field in sensitive_fields:
            if field in masked:
                value = str(masked[field])
                if len(value) > 4:
                    masked[field] = '*' * (len(value) - 4) + value[-4:]
                else:
                    masked[field] = '*' * len(value)
                    
        return masked
        
class RateLimiter:
    """Rate limiter for API endpoints."""
    
    def __init__(
        self,
        max_requests: int = 100,
        time_window: int = 3600
    ) -> None:
        """Initialize rate limiter.
        
        Args:
            max_requests: Maximum number of requests allowed in time window
            time_window: Time window in seconds
        """
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests: Dict[str, list] = {}
        
    def is_allowed(self, client_id: str) -> bool:
        """Check if request is allowed for client.
        
        Args:
            client_id: Client identifier
            
        Returns:
            True if request is allowed
        """
        now = datetime.now().timestamp()
        
        # Initialize client's request history
        if client_id not in self.requests:
            self.requests[client_id] = []
            
        # Remove old requests
        self.requests[client_id] = [
            timestamp for timestamp in self.requests[client_id]
            if now - timestamp <= self.time_window
        ]
        
        # Check if under limit
        if len(self.requests[client_id]) < self.max_requests:
            self.requests[client_id].append(now)
            return True
            
        return False
        
    def get_remaining_requests(self, client_id: str) -> int:
        """Get remaining allowed requests for client.
        
        Args:
            client_id: Client identifier
            
        Returns:
            Number of remaining allowed requests
        """
        if client_id not in self.requests:
            return self.max_requests
            
        now = datetime.now().timestamp()
        valid_requests = [
            timestamp for timestamp in self.requests[client_id]
            if now - timestamp <= self.time_window
        ]
        
        return max(0, self.max_requests - len(valid_requests))
        
class APIKeyManager:
    """Manager for API key operations."""
    
    def __init__(self, keys_file: Optional[Path] = None) -> None:
        """Initialize API key manager.
        
        Args:
            keys_file: Optional path to keys file
        """
        self.keys_file = keys_file
        self.api_keys: Dict[str, Dict] = {}
        
        if keys_file and keys_file.exists():
            self._load_keys()
            
    def _load_keys(self) -> None:
        """Load API keys from file."""
        try:
            with open(self.keys_file, 'r') as f:
                self.api_keys = json.load(f)
        except Exception as e:
            logger.error(f"Error loading API keys: {e}")
            
    def _save_keys(self) -> None:
        """Save API keys to file."""
        if not self.keys_file:
            return
            
        try:
            with open(self.keys_file, 'w') as f:
                json.dump(self.api_keys, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving API keys: {e}")
            
    def generate_api_key(
        self,
        client_id: str,
        permissions: Optional[List[str]] = None
    ) -> str:
        """Generate new API key for client.
        
        Args:
            client_id: Client identifier
            permissions: Optional list of permissions
            
        Returns:
            Generated API key
        """
        # Generate random key
        api_key = base64.b64encode(os.urandom(32)).decode('utf-8')
        
        # Store key info
        self.api_keys[api_key] = {
            'client_id': client_id,
            'permissions': permissions or [],
            'created_at': datetime.now().isoformat(),
            'last_used': None
        }
        
        self._save_keys()
        return api_key
        
    def validate_api_key(
        self,
        api_key: str,
        required_permissions: Optional[List[str]] = None
    ) -> bool:
        """Validate API key and permissions.
        
        Args:
            api_key: API key to validate
            required_permissions: Optional required permissions
            
        Returns:
            True if key is valid and has required permissions
        """
        if api_key not in self.api_keys:
            return False
            
        # Update last used timestamp
        self.api_keys[api_key]['last_used'] = datetime.now().isoformat()
        self._save_keys()
        
        # Check permissions if required
        if required_permissions:
            key_permissions = set(self.api_keys[api_key]['permissions'])
            return all(perm in key_permissions for perm in required_permissions)
            
        return True
        
    def revoke_api_key(self, api_key: str) -> None:
        """Revoke an API key.
        
        Args:
            api_key: API key to revoke
        """
        if api_key in self.api_keys:
            del self.api_keys[api_key]
            self._save_keys()
            
    def get_key_info(self, api_key: str) -> Optional[Dict]:
        """Get information about an API key.
        
        Args:
            api_key: API key to look up
            
        Returns:
            Key information dictionary or None if not found
        """
        return self.api_keys.get(api_key) 