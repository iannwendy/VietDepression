# Authentication System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add user authentication (login/register) with admin/user roles. Each user has isolated prediction history. Admin gets dedicated dashboard.

**Architecture:**

- Backend: FastAPI + SQLite, JWT tokens (httpOnly cookies), bcrypt password hashing
- Frontend: React 18 + TypeScript, protected routes, auth context, TailwindCSS
- Auth flow: JWT stored in httpOnly cookies, auto-refresh pattern, role-based access

## Global Constraints

- **Theme colors**: Primary `#0D9488`, Accent `#F97316`, Background `#F8FAFC`
- **Fonts**: Display `Fraunces`, Body `Manrope`, Mono `JetBrains Mono`
- **Users**: admin/admin123 (role: admin), user/user123 (role: user)
- **Database**: SQLite at `web_demo/backend/data/predictions.db`
- **JWT expiry**: 24 hours
- **API base**: `/api`

---

## Backend Tasks

### Task 1: Backend - Auth Dependencies &amp; Database Schema

**Files:**

- Modify: `web_demo/backend/requirements.txt`
- Create: `web_demo/backend/auth.py`
- Modify: `web_demo/backend/database.py`

**Interfaces:**

- Produces: `auth.py` with `hash_password()`, `verify_password()`, `create_access_token()`, `get_current_user()`, `require_role()`
- Produces: `database.py` updated with `users` table and user CRUD functions

```python
# New Pydantic models in auth.py
class UserCreate(BaseModel): username, password
class UserLogin(BaseModel): username, password
class UserResponse(BaseModel): id, username, role, created_at
class Token(BaseModel): access_token, token_type
```

- [ ] **Step 1: Add dependencies to requirements.txt**

```txt
# Add to web_demo/backend/requirements.txt
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4
python-multipart==0.0.9
```

- [ ] **Step 2: Create auth.py with password hashing and JWT**

```python
# web_demo/backend/auth.py
from datetime import datetime, timedelta
from typing import Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel

SECRET_KEY = "your-secret-key-change-in-production-minimum-32-chars"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_HOURS = 24

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/auth/login")

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None
    role: Optional[str] = None

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def decode_token(token: str) -> TokenData:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        role: str = payload.get("role")
        return TokenData(username=username, role=role)
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

async def get_current_user(token: str = Depends(oauth2_scheme)) -> dict:
    token_data = decode_token(token)
    if token_data.username is None:
        raise HTTPException(status_code=401, detail="Invalid token")
    from database import get_user_by_username
    user = get_user_by_username(token_data.username)
    if user is None:
        raise HTTPException(status_code=401, detail="User not found")
    return user

def require_role(required_role: str):
    async def role_checker(current_user: dict = Depends(get_current_user)):
        if current_user.get("role") != required_role:
            raise HTTPException(status_code=403, detail="Insufficient permissions")
        return current_user
    return role_checker
```

- [ ] **Step 3: Update database.py with users table and CRUD**

Add after line 43 (after _CREATE_INDEX):

```python
_CREATE_USERS_TABLE = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'user' CHECK (role IN ('admin', 'user')),
    is_active INTEGER DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

_CREATE_PREDICTIONS_USER_FK = """
ALTER TABLE predictions ADD COLUMN user_id INTEGER REFERENCES users(id);
"""

_CREATE_USERS_INDEX = """
CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
"""

_CREATE_PREDICTIONS_USER_INDEX = """
CREATE INDEX IF NOT EXISTS idx_predictions_user_id ON predictions(user_id);
"""
```

Add new functions after init_db():

```python
def init_users_table() -> None:
    """Initialize users table and create default admin/user."""
    conn = get_connection()
    try:
        conn.executescript(_CREATE_USERS_TABLE)
        conn.execute(_CREATE_USERS_INDEX)
        conn.execute(_CREATE_PREDICTIONS_USER_INDEX)

        # Check if default users exist
        admin = conn.execute("SELECT id FROM users WHERE username = ?", ("admin",)).fetchone()
        if not admin:
            from auth import get_password_hash
            conn.execute(
                "INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
                ("admin", get_password_hash("admin123"), "admin")
            )
            conn.execute(
                "INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
                ("user", get_password_hash("user123"), "user")
            )
            conn.commit()
            logger.info("Default users created: admin, user")
    finally:
        conn.close()

def get_user_by_username(username: str) -> Optional[dict]:
    """Fetch user by username."""
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT id, username, password_hash, role, is_active, created_at FROM users WHERE username = ? AND is_active = 1",
            (username,)
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()

def get_user_by_id(user_id: int) -> Optional[dict]:
    """Fetch user by ID."""
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT id, username, role, is_active, created_at FROM users WHERE id = ? AND is_active = 1",
            (user_id,)
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()

def create_user(username: str, password_hash: str) -> Optional[dict]:
    """Create a new user."""
    conn = get_connection()
    try:
        cur = conn.execute(
            "INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
            (username, password_hash, "user")
        )
        conn.commit()
        return get_user_by_id(cur.lastrowid)
    except sqlite3.IntegrityError:
        return None
    finally:
        conn.close()

def get_all_users() -> list[dict]:
    """Get all users (for admin)."""
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT id, username, role, is_active, created_at FROM users ORDER BY created_at DESC"
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()
```

Update init_db() function to also call init_users_table():

```python
def init_db() -> None:
    conn = get_connection()
    try:
        conn.executescript(_CREATE_TABLE)
        conn.executescript(_CREATE_INDEX)
        init_users_table()  # Add this line
        conn.commit()
        logger.info("Database initialized")
    finally:
        conn.close()
```

Update save_prediction() to accept user_id:

```python
def save_prediction(
    text: str,
    prediction: str,
    confidence: float,
    *,
    prob_normal: float = 0.0,
    prob_depression: float = 0.0,
    topic_id: Optional[int] = None,
    topic_name: Optional[str] = None,
    risk_level: str = "low",
    model_name: Optional[str] = None,
    user_id: Optional[int] = None,  # Add this
) -> dict:
    conn = get_connection()
    try:
        cur = conn.execute(
            """
            INSERT INTO predictions
                (id, text, prediction, confidence, prob_normal, prob_depression,
                 topic_id, topic_name, risk_level, model_name, user_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(uuid.uuid4()),
                text,
                prediction,
                confidence,
                prob_normal,
                prob_depression,
                topic_id,
                topic_name,
                risk_level,
                model_name,
                user_id,  # Add this
            ),
        )
        conn.commit()
        row_id = cur.lastrowid
        row = conn.execute(
            "SELECT * FROM predictions WHERE rowid = ?", (row_id,)
        ).fetchone()
        return dict(row) if row else {}
    finally:
        conn.close()
```

Update get_history() to accept user_id:

```python
def get_history(limit: int = 50, offset: int = 0, user_id: Optional[int] = None) -> list[dict]:
    conn = get_connection()
    try:
        if user_id:
            rows = conn.execute(
                """
                SELECT * FROM predictions
                WHERE user_id = ?
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
                """,
                (user_id, limit, offset),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT * FROM predictions
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
                """,
                (limit, offset),
            ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()
```

Update get_history_count() to accept user_id:

```python
def get_history_count(user_id: Optional[int] = None) -> int:
    conn = get_connection()
    try:
        if user_id:
            row = conn.execute(
                "SELECT COUNT(*) as cnt FROM predictions WHERE user_id = ?", (user_id,)
            ).fetchone()
        else:
            row = conn.execute("SELECT COUNT(*) as cnt FROM predictions").fetchone()
        return row["cnt"] if row else 0
    finally:
        conn.close()
```

Add new stats functions for admin:

```python
def get_admin_stats() -> dict:
    """Get stats for admin dashboard."""
    conn = get_connection()
    try:
        total_users = conn.execute("SELECT COUNT(*) as cnt FROM users WHERE is_active = 1").fetchone()["cnt"]
        total_predictions = conn.execute("SELECT COUNT(*) as cnt FROM predictions").fetchone()["cnt"]
        predictions_by_user = conn.execute(
            """
            SELECT u.username, COUNT(p.id) as pred_count
            FROM users u
            LEFT JOIN predictions p ON u.id = p.user_id
            GROUP BY u.id, u.username
            ORDER BY pred_count DESC
            """,
        ).fetchall()
        recent_predictions = conn.execute(
            """
            SELECT p.*, u.username
            FROM predictions p
            LEFT JOIN users u ON p.user_id = u.id
            ORDER BY p.created_at DESC
            LIMIT 10
            """,
        ).fetchall()
        return {
            "total_users": total_users,
            "total_predictions": total_predictions,
            "predictions_by_user": [dict(row) for row in predictions_by_user],
            "recent_predictions": [dict(row) for row in recent_predictions],
        }
    finally:
        conn.close()
```

- [ ] **Step 4: Update main.py with auth endpoints**

Add imports after existing imports:

```python
from auth import (
    Token, create_access_token, get_password_hash, verify_password,
    get_current_user, require_role, decode_token, get_user_by_username, create_user
)
from database import get_user_by_id, get_all_users, get_admin_stats, get_user_by_username as db_get_user
```

Add Pydantic models for auth:

```python
class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=30)
    password: str = Field(..., min_length=6)

class UserLogin(BaseModel):
    username: str
    password: str

class UserResponse(BaseModel):
    id: int
    username: str
    role: str
    created_at: str

class AuthResponse(BaseModel):
    user: UserResponse
    access_token: str
    token_type: str
```

Add auth endpoints before "# ── Dashboard Stats ────────────────────────────────────────────────────────────":

```python
# ── Auth Endpoints ────────────────────────────────────────────────────────────

@app.post("/api/auth/register", response_model=AuthResponse)
async def register(request: UserCreate):
    """Register a new user."""
    existing = get_user_by_username(request.username)
    if existing:
        raise HTTPException(status_code=400, detail="Username already exists")

    user = create_user(request.username, get_password_hash(request.password))
    if not user:
        raise HTTPException(status_code=400, detail="Failed to create user")

    access_token = create_access_token(
        data={"sub": user["username"], "role": user["role"]}
    )

    return AuthResponse(
        user=UserResponse(
            id=user["id"],
            username=user["username"],
            role=user["role"],
            created_at=user["created_at"],
        ),
        access_token=access_token,
        token_type="bearer",
    )


@app.post("/api/auth/login", response_model=AuthResponse)
async def login(request: UserLogin):
    """Login and get access token."""
    user = get_user_by_username(request.username)
    if not user or not verify_password(request.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid username or password")

    access_token = create_access_token(
        data={"sub": user["username"], "role": user["role"]}
    )

    return AuthResponse(
        user=UserResponse(
            id=user["id"],
            username=user["username"],
            role=user["role"],
            created_at=user["created_at"],
        ),
        access_token=access_token,
        token_type="bearer",
    )


@app.get("/api/auth/me", response_model=UserResponse)
async def get_me(current_user: dict = Depends(get_current_user)):
    """Get current user info."""
    return UserResponse(
        id=current_user["id"],
        username=current_user["username"],
        role=current_user["role"],
        created_at=current_user["created_at"],
    )


@app.post("/api/auth/logout")
async def logout():
    """Logout (client should delete token)."""
    return {"message": "Logged out successfully"}


# ── Admin Endpoints ──────────────────────────────────────────────────────────

@app.get("/api/admin/stats")
async def get_admin_dashboard_stats(current_user: dict = Depends(require_role("admin"))):
    """Get admin dashboard stats."""
    return get_admin_stats()


@app.get("/api/admin/users")
async def get_users(current_user: dict = Depends(require_role("admin"))):
    """Get all users (admin only)."""
    users = get_all_users()
    return {"users": [UserResponse(**u) for u in users]}
```

Update prediction endpoint to save user_id:

```python
@app.post("/api/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest, current_user: dict = Depends(get_current_user)):
    # ... existing code ...
    try:
        import database
        database.save_prediction(
            text=request.text,
            prediction=prediction,
            confidence=confidence,
            prob_normal=pred_result.get("prob_normal", 0),
            prob_depression=pred_result.get("prob_depression", confidence),
            topic_id=topic_info.get("topic_id") if topic_info else None,
            topic_name=topic_info.get("topic_name") if topic_info else None,
            risk_level=risk_level,
            model_name=model_name,
            user_id=current_user.get("id"),  # Add this
        )
    except Exception as e:
        logger.warning("Failed to save to history: %s", e)
    # ... rest of code ...
```

Update history endpoint to filter by user:

```python
@app.get("/api/history", response_model=HistoryListResponse)
async def get_history(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    current_user: dict = Depends(get_current_user),
):
    try:
        import database
        # Filter by user_id if not admin
        user_id = None if current_user.get("role") == "admin" else current_user.get("id")
        items = database.get_history(limit=limit, offset=offset, user_id=user_id)
        total = database.get_history_count(user_id=user_id)
        return HistoryListResponse(
            items=[
                HistoryEntry(
                    id=item["id"],
                    text=item["text"],
                    prediction=item["prediction"],
                    confidence=item["confidence"],
                    topic=item.get("topic_name"),
                    risk_level=item["risk_level"],
                    model_name=item.get("model_name"),
                    created_at=item["created_at"],
                )
                for item in items
            ],
            total=total,
            limit=limit,
            offset=offset,
        )
    except Exception as e:
        logger.error("History endpoint failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
```

Update save_history_entry similarly.

- [ ] **Step 5: Test backend auth endpoints**

Run: `cd web_demo/backend && python -c "import auth; print('Auth module OK')"`

Expected: No errors

- [ ] **Step 6: Commit backend changes**

```bash
git add web_demo/backend/requirements.txt web_demo/backend/auth.py web_demo/backend/database.py web_demo/backend/main.py
git commit -m "feat: add authentication system with JWT, users table, admin/user roles"
```

---

### Task 2: Backend - CORS &amp; Cookie Configuration

**Files:**

- Modify: `web_demo/backend/main.py`

**Context:**

- Current CORS: `allow_origins=["*"]` allows all origins
- Need to support credentials for cookies
- Vite dev server runs on port 3000, backend on 8000

- [ ] **Step 1: Update CORS middleware for credentials**

In main.py, update the CORSMiddleware:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],  # Vite dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

- [ ] **Step 2: Commit**

```bash
git add web_demo/backend/main.py
git commit -m "fix: update CORS for credentials support"
```

---

## Frontend Tasks

### Task 3: Frontend - Auth Types &amp; API

**Files:**

- Modify: `web_demo/src/types/index.ts`
- Modify: `web_demo/src/services/api.ts`

**Interfaces:**

- Consumes: Backend auth endpoints
- Produces: Auth-related TypeScript types and API functions

- [ ] **Step 1: Add auth types to index.ts**

Add at the end of the file:

```typescript
export interface User {
  id: number;
  username: string;
  role: 'admin' | 'user';
  created_at: string;
}

export interface AuthResponse {
  user: User;
  access_token: string;
  token_type: string;
}

export interface LoginCredentials {
  username: string;
  password: string;
}

export interface RegisterCredentials {
  username: string;
  password: string;
}

export interface AdminStats {
  total_users: number;
  total_predictions: number;
  predictions_by_user: Array<{
    username: string;
    pred_count: number;
  }>;
  recent_predictions: Array<{
    id: string;
    text: string;
    prediction: string;
    confidence: number;
    username?: string;
    created_at: string;
  }>;
}
```

- [ ] **Step 2: Add auth API functions to api.ts**

Add at the end of api.ts:

```typescript
// ── Auth ─────────────────────────────────────────────────────────────────────

const TOKEN_KEY = 'auth_token';
const USER_KEY = 'auth_user';

export function getToken(): string | null {
  return localStorage.getItem(TOKEN_KEY);
}

export function setToken(token: string): void {
  localStorage.setItem(TOKEN_KEY, token);
}

export function removeToken(): void {
  localStorage.removeItem(TOKEN_KEY);
  localStorage.removeItem(USER_KEY);
}

export function getStoredUser(): User | null {
  const stored = localStorage.getItem(USER_KEY);
  if (!stored) return null;
  try {
    return JSON.parse(stored) as User;
  } catch {
    return null;
  }
}

export function setStoredUser(user: User): void {
  localStorage.setItem(USER_KEY, JSON.stringify(user));
}

function authHeaders(): Record<string, string> {
  const token = getToken();
  return token ? { Authorization: `Bearer ${token}` } : {};
}

export async function login(credentials: LoginCredentials): Promise<AuthResponse> {
  const response = await fetch(`${API_BASE}/auth/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(credentials),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Login failed' }));
    throw new Error(error.detail || 'Login failed');
  }

  const data = await response.json();
  setToken(data.access_token);
  setStoredUser(data.user);
  return data;
}

export async function register(credentials: RegisterCredentials): Promise<AuthResponse> {
  const response = await fetch(`${API_BASE}/auth/register`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(credentials),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Registration failed' }));
    throw new Error(error.detail || 'Registration failed');
  }

  const data = await response.json();
  setToken(data.access_token);
  setStoredUser(data.user);
  return data;
}

export async function logout(): Promise<void> {
  try {
    await fetch(`${API_BASE}/auth/logout`, {
      method: 'POST',
      headers: authHeaders(),
    });
  } finally {
    removeToken();
  }
}

export async function getCurrentUser(): Promise<User> {
  const response = await fetch(`${API_BASE}/auth/me`, {
    headers: { ...authHeaders(), 'Content-Type': 'application/json' },
  });

  if (!response.ok) {
    removeToken();
    throw new Error('Not authenticated');
  }

  const user = await response.json();
  setStoredUser(user);
  return user;
}

// ── Admin ─────────────────────────────────────────────────────────────────────

export async function getAdminStats(): Promise<AdminStats> {
  return fetchJSON<AdminStats>(`${API_BASE}/admin/stats`, {
    headers: authHeaders(),
  });
}

export async function getAdminUsers(): Promise<{ users: User[] }> {
  return fetchJSON<{ users: User[] }>(`${API_BASE}/admin/users`, {
    headers: authHeaders(),
  });
}
```

- [ ] **Step 3: Commit**

```bash
git add web_demo/src/types/index.ts web_demo/src/services/api.ts
git commit -m "feat: add auth types and API functions"
```

---

### Task 4: Frontend - Auth Context &amp; Protected Routes

**Files:**

- Create: `web_demo/src/contexts/AuthContext.tsx`
- Modify: `web_demo/src/App.tsx`

**Interfaces:**

- Consumes: Auth API functions
- Produces: AuthContext with user, login, logout, isAdmin, isLoading

- [ ] **Step 1: Create AuthContext**

Create directory and file: `web_demo/src/contexts/AuthContext.tsx`

```typescript
import { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import { User, getStoredUser, getCurrentUser, login as apiLogin, logout as apiLogout, LoginCredentials } from '@/services/api';

interface AuthContextType {
  user: User | null;
  isLoading: boolean;
  isAuthenticated: boolean;
  isAdmin: boolean;
  login: (credentials: LoginCredentials) => Promise<void>;
  logout: () => Promise<void>;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    // Check for stored user on mount
    const stored = getStoredUser();
    if (stored) {
      setUser(stored);
      // Verify token is still valid
      getCurrentUser()
        .then((freshUser) => setUser(freshUser))
        .catch(() => {
          setUser(null);
        })
        .finally(() => setIsLoading(false));
    } else {
      setIsLoading(false);
    }
  }, []);

  const login = async (credentials: LoginCredentials) => {
    const response = await apiLogin(credentials);
    setUser(response.user);
  };

  const logout = async () => {
    await apiLogout();
    setUser(null);
  };

  const value: AuthContextType = {
    user,
    isLoading,
    isAuthenticated: !!user,
    isAdmin: user?.role === 'admin',
    login,
    logout,
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}
```

- [ ] **Step 2: Create protected route components**

Create: `web_demo/src/components/auth/ProtectedRoute.tsx`

```typescript
import { Navigate, useLocation } from 'react-router-dom';
import { useAuth } from '@/contexts/AuthContext';

interface ProtectedRouteProps {
  children: React.ReactNode;
  requireAdmin?: boolean;
}

export function ProtectedRoute({ children, requireAdmin = false }: ProtectedRouteProps) {
  const { isAuthenticated, isAdmin, isLoading } = useAuth();
  const location = useLocation();

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-background">
        <div className="animate-spin rounded-full h-12 w-12 border-4 border-primary border-t-transparent"></div>
      </div>
    );
  }

  if (!isAuthenticated) {
    return <Navigate to="/login" state={{ from: location }} replace />;
  }

  if (requireAdmin && !isAdmin) {
    return <Navigate to="/dashboard" replace />;
  }

  return <>{children}</>;
}

export function PublicRoute({ children }: { children: React.ReactNode }) {
  const { isAuthenticated, isLoading } = useAuth();

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-background">
        <div className="animate-spin rounded-full h-12 w-12 border-4 border-primary border-t-transparent"></div>
      </div>
    );
  }

  if (isAuthenticated) {
    return <Navigate to="/" replace />;
  }

  return <>{children}</>;
}
```

- [ ] **Step 3: Update App.tsx with routes**

```typescript
import './i18n';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { Layout } from '@/components/layout/Layout';
import { AuthProvider } from '@/contexts/AuthContext';
import { ProtectedRoute, PublicRoute } from '@/components/auth/ProtectedRoute';
import Dashboard from '@/pages/Dashboard';
import Prediction from '@/pages/Prediction';
import BatchPrediction from '@/pages/BatchPrediction';
import History from '@/pages/History';
import Login from '@/pages/Login';
import Register from '@/pages/Register';
import Profile from '@/pages/Profile';
import AdminDashboard from '@/pages/AdminDashboard';

function App() {
  return (
    <AuthProvider>
      <BrowserRouter>
        <Routes>
          {/* Public routes */}
          <Route
            path="/login"
            element={
              <PublicRoute>
                <Login />
              </PublicRoute>
            }
          />
          <Route
            path="/register"
            element={
              <PublicRoute>
                <Register />
              </PublicRoute>
            }
          />

          {/* Protected routes */}
          <Route
            path="/"
            element={
              <ProtectedRoute>
                <Layout />
              </ProtectedRoute>
            }
          >
            <Route index element={<Dashboard />} />
            <Route path="prediction" element={<Prediction />} />
            <Route path="batch" element={<BatchPrediction />} />
            <Route path="history" element={<History />} />
            <Route path="profile" element={<Profile />} />
            {/* Hidden routes */}
            <Route path="topics" element={<Navigate to="/" replace />} />
            <Route path="statistics" element={<Navigate to="/" replace />} />
            <Route path="compare" element={<Navigate to="/" replace />} />
          </Route>

          {/* Admin routes */}
          <Route
            path="/admin"
            element={
              <ProtectedRoute requireAdmin>
                <Layout />
              </ProtectedRoute>
            }
          >
            <Route index element={<AdminDashboard />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </AuthProvider>
  );
}

export default App;
```

- [ ] **Step 4: Commit**

```bash
git add web_demo/src/contexts/AuthContext.tsx web_demo/src/components/auth/ProtectedRoute.tsx web_demo/src/App.tsx
git commit -m "feat: add auth context and protected routes"
```

---

### Task 5: Frontend - Login &amp; Register Pages (ProMAX UI)

**Files:**

- Create: `web_demo/src/pages/Login.tsx`
- Create: `web_demo/src/pages/Register.tsx`

**Context:**

- Use TailwindCSS with existing theme colors
- Match existing UI patterns in the codebase
- ProMAX design: clean, modern, accessible

**Design Specs:**

- Centered card on gradient background
- Logo/brand at top
- Form with floating labels
- Loading state on submit
- Error messages inline
- Link to switch between login/register

- [ ] **Step 1: Create Login page**

```typescript
// web_demo/src/pages/Login.tsx
import { useState, FormEvent } from 'react';
import { Link, useNavigate, useLocation } from 'react-router-dom';
import { Brain, Eye, EyeOff, Loader2 } from 'lucide-react';
import { useAuth } from '@/contexts/AuthContext';

export default function Login() {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const { login } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();
  const from = (location.state as { from?: Location })?.from?.pathname || '/';

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setError('');
    setIsLoading(true);

    try {
      await login({ username, password });
      navigate(from, { replace: true });
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Login failed');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-background to-teal-50 flex items-center justify-center p-4">
      <div className="w-full max-w-md">
        {/* Logo */}
        <div className="text-center mb-8">
          <div className="inline-flex items-center justify-center w-16 h-16 rounded-2xl bg-gradient-to-br from-primary to-primary-light shadow-xl shadow-primary/30 mb-4">
            <Brain className="w-8 h-8 text-white" />
          </div>
          <h1 className="font-display text-3xl font-bold text-dark mb-2">
            Mental Health AI
          </h1>
          <p className="text-muted">
            Depression Detection Platform
          </p>
        </div>

        {/* Card */}
        <div className="bg-white rounded-2xl shadow-xl shadow-slate-200/50 p-8 border border-slate-100">
          <h2 className="text-xl font-semibold text-dark mb-6 text-center">
            Welcome back
          </h2>

          <form onSubmit={handleSubmit} className="space-y-5">
            {/* Username */}
            <div>
              <label htmlFor="username" className="block text-sm font-medium text-dark mb-2">
                Username
              </label>
              <input
                id="username"
                type="text"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                className="w-full px-4 py-3 rounded-xl border border-slate-200 focus:border-primary focus:ring-2 focus:ring-primary/20 outline-none transition-all"
                placeholder="Enter your username"
                required
                disabled={isLoading}
              />
            </div>

            {/* Password */}
            <div>
              <label htmlFor="password" className="block text-sm font-medium text-dark mb-2">
                Password
              </label>
              <div className="relative">
                <input
                  id="password"
                  type={showPassword ? 'text' : 'password'}
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  className="w-full px-4 py-3 pr-12 rounded-xl border border-slate-200 focus:border-primary focus:ring-2 focus:ring-primary/20 outline-none transition-all"
                  placeholder="Enter your password"
                  required
                  disabled={isLoading}
                />
                <button
                  type="button"
                  onClick={() => setShowPassword(!showPassword)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-muted hover:text-dark transition-colors"
                >
                  {showPassword ? <EyeOff className="w-5 h-5" /> : <Eye className="w-5 h-5" />}
                </button>
              </div>
            </div>

            {/* Error */}
            {error && (
              <div className="p-3 rounded-xl bg-red-50 border border-red-200 text-red-600 text-sm">
                {error}
              </div>
            )}

            {/* Submit */}
            <button
              type="submit"
              disabled={isLoading}
              className="w-full py-3 px-4 rounded-xl bg-primary text-white font-semibold shadow-lg shadow-primary/30 hover:bg-primary-dark transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
            >
              {isLoading ? (
                <>
                  <Loader2 className="w-5 h-5 animate-spin" />
                  Signing in...
                </>
              ) : (
                'Sign In'
              )}
            </button>
          </form>

          {/* Register link */}
          <p className="mt-6 text-center text-sm text-muted">
            Don't have an account?{' '}
            <Link to="/register" className="text-primary font-medium hover:underline">
              Create one
            </Link>
          </p>
        </div>

        {/* Demo credentials */}
        <div className="mt-6 p-4 bg-white/50 rounded-xl border border-slate-200">
          <p className="text-xs text-muted text-center mb-2">Demo Credentials</p>
          <div className="grid grid-cols-2 gap-2 text-xs">
            <div className="p-2 bg-slate-100 rounded-lg">
              <p className="font-medium text-dark">Admin</p>
              <p className="text-muted">admin / admin123</p>
            </div>
            <div className="p-2 bg-slate-100 rounded-lg">
              <p className="font-medium text-dark">User</p>
              <p className="text-muted">user / user123</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Create Register page**

```typescript
// web_demo/src/pages/Register.tsx
import { useState, FormEvent } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { Brain, Eye, EyeOff, Loader2, Check } from 'lucide-react';
import { useAuth } from '@/contexts/AuthContext';

export default function Register() {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const { login } = useAuth();
  const navigate = useNavigate();

  const passwordRequirements = [
    { met: password.length >= 6, text: 'At least 6 characters' },
    { met: password === confirmPassword && password.length > 0, text: 'Passwords match' },
  ];

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setError('');

    if (password !== confirmPassword) {
      setError('Passwords do not match');
      return;
    }

    if (password.length < 6) {
      setError('Password must be at least 6 characters');
      return;
    }

    setIsLoading(true);

    try {
      await login({ username, password }); // login after register
      navigate('/', { replace: true });
    } catch (err) {
      // If login fails after register, navigate to login
      if (err instanceof Error && err.message.includes('Username already exists')) {
        setError('Username already exists. Please choose another.');
      } else {
        setError(err instanceof Error ? err.message : 'Registration failed');
      }
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-background to-teal-50 flex items-center justify-center p-4">
      <div className="w-full max-w-md">
        {/* Logo */}
        <div className="text-center mb-8">
          <div className="inline-flex items-center justify-center w-16 h-16 rounded-2xl bg-gradient-to-br from-primary to-primary-light shadow-xl shadow-primary/30 mb-4">
            <Brain className="w-8 h-8 text-white" />
          </div>
          <h1 className="font-display text-3xl font-bold text-dark mb-2">
            Mental Health AI
          </h1>
          <p className="text-muted">
            Create your account
          </p>
        </div>

        {/* Card */}
        <div className="bg-white rounded-2xl shadow-xl shadow-slate-200/50 p-8 border border-slate-100">
          <h2 className="text-xl font-semibold text-dark mb-6 text-center">
            Create Account
          </h2>

          <form onSubmit={handleSubmit} className="space-y-5">
            {/* Username */}
            <div>
              <label htmlFor="username" className="block text-sm font-medium text-dark mb-2">
                Username
              </label>
              <input
                id="username"
                type="text"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                className="w-full px-4 py-3 rounded-xl border border-slate-200 focus:border-primary focus:ring-2 focus:ring-primary/20 outline-none transition-all"
                placeholder="Choose a username"
                required
                minLength={3}
                maxLength={30}
                disabled={isLoading}
              />
            </div>

            {/* Password */}
            <div>
              <label htmlFor="password" className="block text-sm font-medium text-dark mb-2">
                Password
              </label>
              <div className="relative">
                <input
                  id="password"
                  type={showPassword ? 'text' : 'password'}
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  className="w-full px-4 py-3 pr-12 rounded-xl border border-slate-200 focus:border-primary focus:ring-2 focus:ring-primary/20 outline-none transition-all"
                  placeholder="Create a password"
                  required
                  minLength={6}
                  disabled={isLoading}
                />
                <button
                  type="button"
                  onClick={() => setShowPassword(!showPassword)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-muted hover:text-dark transition-colors"
                >
                  {showPassword ? <EyeOff className="w-5 h-5" /> : <Eye className="w-5 h-5" />}
                </button>
              </div>

              {/* Password requirements */}
              {password.length > 0 && (
                <div className="mt-2 space-y-1">
                  {passwordRequirements.map((req, i) => (
                    <div key={i} className={`flex items-center gap-2 text-xs ${req.met ? 'text-green-600' : 'text-muted'}`}>
                      <Check className={`w-3 h-3 ${req.met ? 'opacity-100' : 'opacity-30'}`} />
                      {req.text}
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Confirm Password */}
            <div>
              <label htmlFor="confirmPassword" className="block text-sm font-medium text-dark mb-2">
                Confirm Password
              </label>
              <input
                id="confirmPassword"
                type={showPassword ? 'text' : 'password'}
                value={confirmPassword}
                onChange={(e) => setConfirmPassword(e.target.value)}
                className="w-full px-4 py-3 rounded-xl border border-slate-200 focus:border-primary focus:ring-2 focus:ring-primary/20 outline-none transition-all"
                placeholder="Confirm your password"
                required
                disabled={isLoading}
              />
            </div>

            {/* Error */}
            {error && (
              <div className="p-3 rounded-xl bg-red-50 border border-red-200 text-red-600 text-sm">
                {error}
              </div>
            )}

            {/* Submit */}
            <button
              type="submit"
              disabled={isLoading || !passwordRequirements.every((r) => r.met)}
              className="w-full py-3 px-4 rounded-xl bg-primary text-white font-semibold shadow-lg shadow-primary/30 hover:bg-primary-dark transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
            >
              {isLoading ? (
                <>
                  <Loader2 className="w-5 h-5 animate-spin" />
                  Creating account...
                </>
              ) : (
                'Create Account'
              )}
            </button>
          </form>

          {/* Login link */}
          <p className="mt-6 text-center text-sm text-muted">
            Already have an account?{' '}
            <Link to="/login" className="text-primary font-medium hover:underline">
              Sign in
            </Link>
          </p>
        </div>
      </div>
    </div>
  );
}
```

- [ ] **Step 3: Commit**

```bash
git add web_demo/src/pages/Login.tsx web_demo/src/pages/Register.tsx
git commit -m "feat: add Login and Register pages with ProMAX UI"
```

---

### Task 6: Frontend - Profile &amp; Admin Dashboard Pages

**Files:**

- Create: `web_demo/src/pages/Profile.tsx`
- Create: `web_demo/src/pages/AdminDashboard.tsx`

- [ ] **Step 1: Create Profile page**

```typescript
// web_demo/src/pages/Profile.tsx
import { useState } from 'react';
import { User, Settings, Shield, LogOut } from 'lucide-react';
import { useAuth } from '@/contexts/AuthContext';
import { useNavigate } from 'react-router-dom';

export default function Profile() {
  const { user, logout, isAdmin } = useAuth();
  const navigate = useNavigate();
  const [isLoggingOut, setIsLoggingOut] = useState(false);

  const handleLogout = async () => {
    setIsLoggingOut(true);
    try {
      await logout();
      navigate('/login', { replace: true });
    } finally {
      setIsLoggingOut(false);
    }
  };

  return (
    <div className="max-w-2xl mx-auto animate-fade-in">
      <h1 className="font-display text-2xl font-bold text-dark mb-8">
        Profile Settings
      </h1>

      {/* User Info Card */}
      <div className="bg-white rounded-2xl shadow-lg border border-slate-100 overflow-hidden mb-6">
        <div className="bg-gradient-to-r from-primary to-primary-light p-6">
          <div className="flex items-center gap-4">
            <div className="w-16 h-16 rounded-full bg-white/20 flex items-center justify-center">
              <User className="w-8 h-8 text-white" />
            </div>
            <div>
              <h2 className="text-xl font-semibold text-white">{user?.username}</h2>
              <p className="text-white/80 text-sm capitalize">{user?.role}</p>
            </div>
          </div>
        </div>

        <div className="p-6 space-y-4">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-slate-100 flex items-center justify-center">
              <User className="w-5 h-5 text-muted" />
            </div>
            <div>
              <p className="text-xs text-muted uppercase tracking-wide">Username</p>
              <p className="font-medium text-dark">{user?.username}</p>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-slate-100 flex items-center justify-center">
              <Shield className="w-5 h-5 text-muted" />
            </div>
            <div>
              <p className="text-xs text-muted uppercase tracking-wide">Role</p>
              <p className="font-medium text-dark capitalize">{user?.role}</p>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-slate-100 flex items-center justify-center">
              <Settings className="w-5 h-5 text-muted" />
            </div>
            <div>
              <p className="text-xs text-muted uppercase tracking-wide">Member Since</p>
              <p className="font-medium text-dark">
                {user?.created_at ? new Date(user.created_at).toLocaleDateString() : 'N/A'}
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Admin Link */}
      {isAdmin && (
        <button
          onClick={() => navigate('/admin')}
          className="w-full p-4 bg-white rounded-xl border border-slate-200 hover:border-primary hover:bg-primary/5 transition-all flex items-center justify-between mb-6"
        >
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-accent/10 flex items-center justify-center">
              <Shield className="w-5 h-5 text-accent" />
            </div>
            <div className="text-left">
              <p className="font-medium text-dark">Admin Dashboard</p>
              <p className="text-sm text-muted">View system statistics and manage users</p>
            </div>
          </div>
          <svg className="w-5 h-5 text-muted" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
          </svg>
        </button>
      )}

      {/* Logout */}
      <button
        onClick={handleLogout}
        disabled={isLoggingOut}
        className="w-full p-4 bg-white rounded-xl border border-red-200 hover:bg-red-50 transition-all flex items-center justify-center gap-2 text-red-600"
      >
        <LogOut className="w-5 h-5" />
        {isLoggingOut ? 'Signing out...' : 'Sign Out'}
      </button>
    </div>
  );
}
```

- [ ] **Step 2: Create Admin Dashboard page**

```typescript
// web_demo/src/pages/AdminDashboard.tsx
import { useState, useEffect } from 'react';
import { Users, MessageSquare, Activity, TrendingUp, Clock, AlertCircle } from 'lucide-react';
import { getAdminStats, getAdminUsers, AdminStats, User } from '@/services/api';

export default function AdminDashboard() {
  const [stats, setStats] = useState<AdminStats | null>(null);
  const [users, setUsers] = useState<User[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    setIsLoading(true);
    setError('');
    try {
      const [statsData, usersData] = await Promise.all([
        getAdminStats(),
        getAdminUsers(),
      ]);
      setStats(statsData);
      setUsers(usersData.users);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load data');
    } finally {
      setIsLoading(false);
    }
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-20">
        <div className="animate-spin rounded-full h-12 w-12 border-4 border-primary border-t-transparent"></div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-6 bg-red-50 border border-red-200 rounded-xl text-red-600">
        <AlertCircle className="w-5 h-5 mb-2" />
        {error}
      </div>
    );
  }

  const statCards = [
    {
      title: 'Total Users',
      value: stats?.total_users || 0,
      icon: Users,
      color: 'text-primary',
      bgColor: 'bg-primary/10',
    },
    {
      title: 'Total Predictions',
      value: stats?.total_predictions || 0,
      icon: MessageSquare,
      color: 'text-accent',
      bgColor: 'bg-accent/10',
    },
  ];

  return (
    <div className="space-y-8 animate-fade-in">
      <h1 className="font-display text-2xl font-bold text-dark">
        Admin Dashboard
      </h1>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {statCards.map((card) => (
          <div
            key={card.title}
            className="bg-white rounded-2xl p-6 shadow-lg border border-slate-100"
          >
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted mb-1">{card.title}</p>
                <p className="text-3xl font-bold text-dark">{card.value}</p>
              </div>
              <div className={`w-12 h-12 rounded-xl ${card.bgColor} flex items-center justify-center`}>
                <card.icon className={`w-6 h-6 ${card.color}`} />
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Users Table */}
      <div className="bg-white rounded-2xl shadow-lg border border-slate-100 overflow-hidden">
        <div className="p-6 border-b border-slate-100">
          <h2 className="text-lg font-semibold text-dark flex items-center gap-2">
            <Users className="w-5 h-5 text-primary" />
            Users
          </h2>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="bg-slate-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-muted uppercase tracking-wider">
                  Username
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-muted uppercase tracking-wider">
                  Role
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-muted uppercase tracking-wider">
                  Created
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {users.map((user) => (
                <tr key={user.id} className="hover:bg-slate-50 transition-colors">
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span className="font-medium text-dark">{user.username}</span>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span
                      className={`px-2 py-1 text-xs font-medium rounded-full ${
                        user.role === 'admin'
                          ? 'bg-accent/10 text-accent'
                          : 'bg-slate-100 text-muted'
                      }`}
                    >
                      {user.role}
                    </span>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-muted">
                    {new Date(user.created_at).toLocaleDateString()}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Predictions by User */}
      <div className="bg-white rounded-2xl shadow-lg border border-slate-100 overflow-hidden">
        <div className="p-6 border-b border-slate-100">
          <h2 className="text-lg font-semibold text-dark flex items-center gap-2">
            <TrendingUp className="w-5 h-5 text-primary" />
            Predictions by User
          </h2>
        </div>
        <div className="p-6">
          {stats?.predictions_by_user && stats.predictions_by_user.length > 0 ? (
            <div className="space-y-3">
              {stats.predictions_by_user.map((item) => (
                <div key={item.username} className="flex items-center justify-between">
                  <span className="font-medium text-dark">{item.username}</span>
                  <span className="px-3 py-1 bg-primary/10 text-primary rounded-full text-sm font-medium">
                    {item.pred_count} predictions
                  </span>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-muted text-center py-4">No predictions yet</p>
          )}
        </div>
      </div>

      {/* Recent Activity */}
      <div className="bg-white rounded-2xl shadow-lg border border-slate-100 overflow-hidden">
        <div className="p-6 border-b border-slate-100">
          <h2 className="text-lg font-semibold text-dark flex items-center gap-2">
            <Clock className="w-5 h-5 text-primary" />
            Recent Activity
          </h2>
        </div>
        <div className="divide-y divide-slate-100">
          {stats?.recent_predictions?.slice(0, 5).map((pred) => (
            <div key={pred.id} className="p-4 hover:bg-slate-50 transition-colors">
              <div className="flex items-start justify-between">
                <div className="flex-1 min-w-0">
                  <p className="text-sm text-dark truncate">{pred.text}</p>
                  <p className="text-xs text-muted mt-1">
                    by {pred.username || 'Anonymous'} • {new Date(pred.created_at).toLocaleString()}
                  </p>
                </div>
                <span
                  className={`ml-4 px-2 py-1 text-xs font-medium rounded-full ${
                    pred.prediction === 'depression'
                      ? 'bg-depression/10 text-depression'
                      : 'bg-normal/10 text-normal'
                  }`}
                >
                  {pred.prediction}
                </span>
              </div>
            </div>
          )) || (
            <p className="p-4 text-muted text-center">No recent activity</p>
          )}
        </div>
      </div>
    </div>
  );
}
```

- [ ] **Step 3: Commit**

```bash
git add web_demo/src/pages/Profile.tsx web_demo/src/pages/AdminDashboard.tsx
git commit -m "feat: add Profile and AdminDashboard pages"
```

---

### Task 7: Frontend - Update Header with User Menu

**Files:**

- Modify: `web_demo/src/components/layout/Header.tsx`

**Context:**

- Show user menu with username, role badge, profile link, logout
- Show/hide based on auth state
- Admin gets link to admin dashboard

- [ ] **Step 1: Update Header with user menu**

```typescript
import { Link, useLocation } from 'react-router-dom';
import { Brain, LayoutDashboard, MessageSquare, Upload, Network, BarChart3, History, GitCompare, User, LogOut, Shield, Menu, X } from 'lucide-react';
import { cn } from '@/lib/utils';
import { LanguageSwitcher } from './LanguageSwitcher';
import { useAuth } from '@/contexts/AuthContext';
import { useState } from 'react';

const navItems = [
  { path: '/', label: 'Dashboard', icon: LayoutDashboard },
  { path: '/prediction', label: 'Prediction', icon: MessageSquare },
  { path: '/batch', label: 'Batch', icon: Upload },
  { path: '/history', label: 'History', icon: History },
  { path: '/profile', label: 'Profile', icon: User },
  // Hidden: Topics, Statistics, Compare (uncomment to enable)
  // { path: '/topics', label: 'Topics', icon: Network },
  // { path: '/statistics', label: 'Statistics', icon: BarChart3 },
  // { path: '/compare', label: 'Compare', icon: GitCompare },
];

export function Header() {
  const location = useLocation();
  const { user, isAdmin, isAuthenticated, logout } = useAuth();
  const [showUserMenu, setShowUserMenu] = useState(false);
  const [showMobileMenu, setShowMobileMenu] = useState(false);

  const handleLogout = async () => {
    await logout();
    setShowUserMenu(false);
  };

  return (
    <header className="sticky top-0 z-50 glass border-b border-slate-200/50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <Link to="/" className="flex items-center gap-3 group">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-primary to-primary-light flex items-center justify-center shadow-lg shadow-primary/25 group-hover:shadow-primary/40 transition-shadow">
              <Brain className="w-5 h-5 text-white" />
            </div>
            <div className="hidden sm:block">
              <h1 className="font-display font-semibold text-dark text-lg leading-tight">
                Mental Health AI
              </h1>
              <p className="text-xs text-muted -mt-0.5">
                Depression Detection Platform
              </p>
            </div>
          </Link>

          {/* Desktop Navigation */}
          {isAuthenticated && (
            <nav className="hidden md:flex items-center gap-1">
              {navItems.map(({ path, label, icon: Icon }) => {
                const isActive = location.pathname === path;
                return (
                  <Link
                    key={path}
                    to={path}
                    className={cn(
                      'flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium transition-all duration-200',
                      isActive
                        ? 'bg-primary text-white shadow-lg shadow-primary/25'
                        : 'text-muted hover:text-dark hover:bg-slate-100'
                    )}
                  >
                    <Icon className="w-4 h-4" />
                    <span>{label}</span>
                  </Link>
                );
              })}
              {isAdmin && (
                <Link
                  to="/admin"
                  className={cn(
                    'flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium transition-all duration-200',
                    location.pathname === '/admin'
                      ? 'bg-accent text-white shadow-lg shadow-accent/25'
                      : 'text-muted hover:text-dark hover:bg-slate-100'
                  )}
                >
                  <Shield className="w-4 h-4" />
                  <span>Admin</span>
                </Link>
              )}
            </nav>
          )}

          {/* Right side */}
          <div className="flex items-center gap-3">
            {/* Language Switcher */}
            {isAuthenticated && <LanguageSwitcher />}

            {/* User Menu (Desktop) */}
            {isAuthenticated ? (
              <div className="relative hidden md:block">
                <button
                  onClick={() => setShowUserMenu(!showUserMenu)}
                  className="flex items-center gap-2 px-3 py-2 rounded-lg hover:bg-slate-100 transition-colors"
                >
                  <div className="w-8 h-8 rounded-full bg-primary flex items-center justify-center">
                    <User className="w-4 h-4 text-white" />
                  </div>
                  <span className="text-sm font-medium text-dark">{user?.username}</span>
                  {isAdmin && (
                    <span className="px-1.5 py-0.5 text-xs font-medium bg-accent/10 text-accent rounded">
                      Admin
                    </span>
                  )}
                </button>

                {showUserMenu && (
                  <>
                    <div
                      className="fixed inset-0"
                      onClick={() => setShowUserMenu(false)}
                    />
                    <div className="absolute right-0 mt-2 w-56 bg-white rounded-xl shadow-xl border border-slate-100 py-2 animate-fade-in">
                      <div className="px-4 py-2 border-b border-slate-100">
                        <p className="font-medium text-dark">{user?.username}</p>
                        <p className="text-xs text-muted capitalize">{user?.role}</p>
                      </div>
                      <Link
                        to="/profile"
                        onClick={() => setShowUserMenu(false)}
                        className="flex items-center gap-2 px-4 py-2 text-sm text-dark hover:bg-slate-50"
                      >
                        <User className="w-4 h-4" />
                        Profile
                      </Link>
                      {isAdmin && (
                        <Link
                          to="/admin"
                          onClick={() => setShowUserMenu(false)}
                          className="flex items-center gap-2 px-4 py-2 text-sm text-dark hover:bg-slate-50"
                        >
                          <Shield className="w-4 h-4" />
                          Admin Dashboard
                        </Link>
                      )}
                      <button
                        onClick={handleLogout}
                        className="flex items-center gap-2 w-full px-4 py-2 text-sm text-red-600 hover:bg-red-50"
                      >
                        <LogOut className="w-4 h-4" />
                        Sign Out
                      </button>
                    </div>
                  </>
                )}
              </div>
            ) : (
              <Link
                to="/login"
                className="px-4 py-2 bg-primary text-white rounded-lg text-sm font-medium hover:bg-primary-dark transition-colors"
              >
                Sign In
              </Link>
            )}

            {/* Mobile Menu Button */}
            <button
              onClick={() => setShowMobileMenu(!showMobileMenu)}
              className="md:hidden p-2 rounded-lg hover:bg-slate-100"
            >
              {showMobileMenu ? (
                <X className="w-5 h-5 text-dark" />
              ) : (
                <Menu className="w-5 h-5 text-dark" />
              )}
            </button>
          </div>
        </div>

        {/* Mobile Navigation */}
        {showMobileMenu && isAuthenticated && (
          <nav className="md:hidden py-4 border-t border-slate-100 animate-slide-up">
            {navItems.map(({ path, label, icon: Icon }) => {
              const isActive = location.pathname === path;
              return (
                <Link
                  key={path}
                  to={path}
                  onClick={() => setShowMobileMenu(false)}
                  className={cn(
                    'flex items-center gap-3 px-4 py-3 rounded-lg text-sm font-medium transition-all',
                    isActive
                      ? 'bg-primary text-white'
                      : 'text-muted hover:text-dark hover:bg-slate-100'
                  )}
                >
                  <Icon className="w-4 h-4" />
                  {label}
                </Link>
              );
            })}
            {isAdmin && (
              <Link
                to="/admin"
                onClick={() => setShowMobileMenu(false)}
                className={cn(
                  'flex items-center gap-3 px-4 py-3 rounded-lg text-sm font-medium transition-all',
                  location.pathname === '/admin'
                    ? 'bg-accent text-white'
                    : 'text-muted hover:text-dark hover:bg-slate-100'
                )}
              >
                <Shield className="w-4 h-4" />
                Admin Dashboard
              </Link>
            )}
          </nav>
        )}
      </div>
    </header>
  );
}
```

- [ ] **Step 2: Commit**

```bash
git add web_demo/src/components/layout/Header.tsx
git commit -m "feat: update Header with user menu and auth state"
```

---

## Integration &amp; Testing

### Task 8: Integration Testing

**Files:**

- Modify: `web_demo/backend/requirements.txt` (install test deps if needed)
- Test manually with curl or browser

- [ ] **Step 1: Verify backend starts without errors**

Run: `cd web_demo/backend && python -c "from main import app; print('Backend OK')"`

Expected: "Backend OK"

- [ ] **Step 2: Verify frontend builds**

Run: `cd web_demo && npm run build 2>&1 | tail -20`

Expected: No TypeScript errors

- [ ] **Step 3: Manual test checklist**

1. Start backend: `cd web_demo/backend && uvicorn main:app --reload --port 8000`
2. Start frontend: `cd web_demo && npm run dev`
3. Visit [http://localhost:5173](http://localhost:5173)
4. Should redirect to /login
5. Login with admin/admin123
6. Should see dashboard with admin menu
7. Click admin link → see admin dashboard
8. Logout → should redirect to login
9. Login with user/user123
10. Should see user dashboard without admin menu

- [ ] **Step 4: Commit final**

```bash
git add -A
git commit -m "feat: complete authentication system - login, register, admin dashboard, user profiles"
```

---

## Summary


| Task | Description                         | Files                                              |
| ---- | ----------------------------------- | -------------------------------------------------- |
| 1    | Backend auth schema &amp; logic     | `auth.py`, `database.py`, `main.py`                |
| 2    | CORS for credentials                | `main.py`                                          |
| 3    | Frontend auth types &amp; API       | `types/index.ts`, `services/api.ts`                |
| 4    | Auth context &amp; protected routes | `AuthContext.tsx`, `ProtectedRoute.tsx`, `App.tsx` |
| 5    | Login &amp; Register pages          | `Login.tsx`, `Register.tsx`                        |
| 6    | Profile &amp; Admin Dashboard       | `Profile.tsx`, `AdminDashboard.tsx`                |
| 7    | Header with user menu               | `Header.tsx`                                       |
| 8    | Integration testing                 | Manual verification                                |


**Total: 8 tasks**

After implementation, verify:

- admin/admin123 → admin dashboard with all stats
- user/user123 → user dashboard with isolated history
- New users can register and login
- Predictions are saved with user_id
- History filters by logged-in user

