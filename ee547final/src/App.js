import React, { useState, useEffect } from 'react';
import './App.css';

// --- 模拟初始数据 (保持原样) ---
const INITIAL_DATA = [
  { id: 1, subject: 'ee547', name: 'hw1', date: '2023-10-01', status: 'graded', score: 95 },
  { id: 2, subject: 'ee641', name: 'hw2', date: '2023-10-05', status: 'graded', score: 88 },
  { id: 3, subject: 'ee541', name: 'hw3', date: '2023-10-10', status: 'pending', score: null }
];

function App() {
  // --- 认证状态管理 ---
  // 尝试从 localStorage 读取 token，如果存在则说明已登录
  const [token, setToken] = useState(localStorage.getItem('access_token'));

  // --- 作业数据状态 ---
  const [homeworkList, setHomeworkList] = useState(INITIAL_DATA);

  // 登录成功回调
  const handleLoginSuccess = (accessToken) => {
    localStorage.setItem('access_token', accessToken);
    setToken(accessToken);
  };

  // 注销回调
  const handleLogout = () => {
    localStorage.removeItem('access_token');
    setToken(null);
  };

  // 添加作业
  const handleAddHomework = (newHomework) => {
    setHomeworkList([newHomework, ...homeworkList]);
  };

  // 如果没有 Token，显示登录/注册页
  if (!token) {
    return <AuthPage onLoginSuccess={handleLoginSuccess} />;
  }

  // 如果有 Token，显示主 Dashboard
  return (
    <div className="container">
      {/* 传递 logout 函数给 Header */}
      <Header onLogout={handleLogout} />

      <div className="main-grid">
        <UploadCard onUpload={handleAddHomework} token={token} />
        <HistoryList homeworks={homeworkList} />
      </div>
    </div>
  );
}

// --- 新增组件：登录/注册页面 ---
const AuthPage = ({ onLoginSuccess }) => {
  const [isLoginView, setIsLoginView] = useState(true); // true: 登录, false: 注册
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  // 表单数据
  const [formData, setFormData] = useState({
    email: '',
    password: '',
    name: '',
    role: 'student' // 默认为 student
  });

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');

    // 注意：这里假设 React 和 FastAPI 运行在不同端口，你可能需要配置 Proxy
    // 或者将 '/api' 替换为 'http://localhost:8000/api'
    const BASE_URL = '/api/v1/auth';

    try {
      if (isLoginView) {
        // --- 登录逻辑 ---
        const response = await fetch(`${BASE_URL}/login`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            email: formData.email,
            password: formData.password
          })
        });

        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || 'Login failed');

        // 登录成功，调用父组件回调
        // 根据后端代码：return Token(access_token=access_token)
        onLoginSuccess(data.access_token);
      } else {
        // --- 注册逻辑 ---
        const response = await fetch(`${BASE_URL}/register`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            email: formData.email,
            password: formData.password,
            name: formData.name,
            role: formData.role
          })
        });

        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || 'Register failed');

        // 注册成功后，自动切换到登录或直接自动登录
        alert('Registration successful! Please log in.');
        setIsLoginView(true);
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="auth-container" style={authStyles.container}>
      <div className="card" style={authStyles.card}>
        <h2>{isLoginView ? 'Login' : 'Register'}</h2>

        {error && <div style={{color: 'red', marginBottom: '10px'}}>{error}</div>}

        <form onSubmit={handleSubmit}>
          <div className="form-group">
            <label>Email</label>
            <input
              name="email"
              type="email"
              required
              value={formData.email}
              onChange={handleChange}
            />
          </div>

          <div className="form-group">
            <label>Password</label>
            <input
              name="password"
              type="password"
              required
              value={formData.password}
              onChange={handleChange}
            />
          </div>

          {/* 注册时额外显示的字段 */}
          {!isLoginView && (
            <>
              <div className="form-group">
                <label>Full Name</label>
                <input
                  name="name"
                  type="text"
                  value={formData.name}
                  onChange={handleChange}
                />
              </div>
              <div className="form-group">
                <label>Role</label>
                <select name="role" value={formData.role} onChange={handleChange}>
                  <option value="student">Student</option>
                  <option value="teacher">Teacher</option>
                </select>
              </div>
            </>
          )}

          <button type="submit" className="btn-submit" disabled={loading} style={{marginTop: '20px'}}>
            {loading ? 'Processing...' : (isLoginView ? 'Login' : 'Register')}
          </button>
        </form>

        <p style={{marginTop: '15px', textAlign: 'center', fontSize: '0.9rem'}}>
          {isLoginView ? "Don't have an account? " : "Already have an account? "}
          <span
            style={{color: 'blue', cursor: 'pointer', textDecoration: 'underline'}}
            onClick={() => { setError(''); setIsLoginView(!isLoginView); }}
          >
            {isLoginView ? 'Register' : 'Login'}
          </span>
        </p>
      </div>
    </div>
  );
};

// --- 简单的内联样式 (仅用于 AuthPage) ---
const authStyles = {
  container: {
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center',
    minHeight: '100vh',
    backgroundColor: '#f5f7fa'
  },
  card: {
    width: '100%',
    maxWidth: '400px',
    padding: '2rem'
  }
};


// --- 修改后的 Header (增加 Logout 按钮) ---
const Header = ({ onLogout }) => (
  <header style={{display: 'flex', justifyContent: 'space-between', alignItems: 'center'}}>
    <h1>Student Homework</h1>
    <div className="user-info" style={{display:'flex', alignItems:'center', gap:'1rem'}}>
      <span>Welcome</span>
      <div className="avatar">User</div>
      <button
        onClick={onLogout}
        style={{
          padding: '5px 10px',
          backgroundColor: '#ff4d4f',
          color: 'white',
          border: 'none',
          borderRadius: '4px',
          cursor: 'pointer'
        }}
      >
        Logout
      </button>
    </div>
  </header>
);

// --- 原有组件保持不变 (UploadCard, HistoryList, StatusBadge) ---
// --- 只是 UploadCard 可以接收 token 以备将来真正上传文件使用 ---

const UploadCard = ({ onUpload, token }) => {
  const [subject, setSubject] = useState('');
  const [name, setName] = useState('');
  const [file, setFile] = useState(null);
  const [isUploading, setIsUploading] = useState(false);

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
    }
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!file || !subject || !name) return alert("please fill correct info");

    setIsUploading(true);

    // 模拟上传 - 在真实场景中，你会在这里使用 fetch 发送 FormData
    // 并带上 Authorization: `Bearer ${token}`
    setTimeout(() => {
      const newEntry = {
        id: Date.now(),
        subject,
        name,
        date: new Date().toISOString().split('T')[0],
        status: 'pending',
        score: null
      };

      onUpload(newEntry);
      setSubject('');
      setName('');
      setFile(null);
      setIsUploading(false);
      alert("success");
    }, 1000);
  };

  return (
    <section className="card">
      <h2>upload new homework</h2>
      <form onSubmit={handleSubmit}>
        <div className="form-group">
          <label>course</label>
          <select
            value={subject}
            onChange={(e) => setSubject(e.target.value)}
            required
          >
            <option value="" disabled>select...</option>
            <option value="ee541">ee541</option>
            <option value="ee503">ee503</option>
            <option value="ee641">ee641</option>
            <option value="ee547">ee547</option>
          </select>
        </div>

        <div className="form-group">
          <label>homework name</label>
          <input
            type="text"
            placeholder=""
            value={name}
            onChange={(e) => setName(e.target.value)}
            required
          />
        </div>

        <div className="form-group">
          <label>file</label>
          <div className="file-drop-area">
            <span style={{ color: file ? 'green' : '#888' }}>
              {file ? `selected: ${file.name}` : 'upload file'}
            </span>
            <input
              type="file"
              className="file-input-hidden"
              onChange={handleFileChange}
              required
            />
          </div>
        </div>

        <button type="submit" className="btn-submit" disabled={isUploading}>
          {isUploading ? 'uploading' : 'submit'}
        </button>
      </form>
    </section>
  );
};

const HistoryList = ({ homeworks }) => {
  return (
    <section className="card">
      <h2>history</h2>
      {homeworks.length === 0 ? (
        <p style={{textAlign:'center', color:'#999'}}>empty</p>
      ) : (
        <table>
          <thead>
            <tr>
              <th>Infomation</th>
              <th>submit time</th>
              <th>status</th>
              <th>score</th>
            </tr>
          </thead>
          <tbody>
            {homeworks.map((item) => (
              <tr key={item.id}>
                <td>
                  <div style={{ fontWeight: 'bold' }}>{item.subject}</div>
                  <div style={{ fontSize: '0.85rem', color: '#666' }}>{item.name}</div>
                </td>
                <td>{item.date}</td>
                <td>
                  <StatusBadge status={item.status} />
                </td>
                <td>
                  {item.status === 'graded' ? (
                    <span style={{ fontWeight: 'bold' }}>{item.score} 分</span>
                  ) : (
                    <span style={{ color: '#ccc' }}>--</span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </section>
  );
};

const StatusBadge = ({ status }) => {
  if (status === 'graded') {
    return <span className="badge badge-graded">Corrected</span>;
  }
  return <span className="badge badge-pending">Pending</span>;
};

export default App;