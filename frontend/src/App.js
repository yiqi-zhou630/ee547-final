import React, { useState, useEffect, useCallback } from 'react';
// 1. 引入必要的路由组件
import { BrowserRouter as Router, Routes, Route, useNavigate } from 'react-router-dom';
import './App.css';
import CreateQuestion from './CreateQuestion';

// --- 1. Header 组件 (修复版：支持路由跳转 + 总是显示Logout) ---
const Header = ({ user, onLogout }) => {
  const navigate = useNavigate();
  // 更加稳健的判断
  const isTeacher = user?.role?.toLowerCase() === 'teacher';

  return (
    <header className="header">
      <h1>EE547 Grading System</h1>
      <div className="header-right">
        {user ? (
          <span className="user-info">{user.name} ({user.role})</span>
        ) : (
          <span className="user-info">...</span>
        )}

        {/* 只有老师显示 Header 上的快捷按钮 */}
        {isTeacher && (
           <button
             className="btn-nav"
             onClick={() => navigate('/create-question')}
           >
             + New Question
           </button>
        )}

        <button onClick={onLogout} className="btn-logout">Logout</button>
      </div>
    </header>
  );
};

// --- 2. 这里的 AuthPage, UploadCard, HistoryList 保持不变 ---

const AuthPage = ({ onLoginSuccess }) => {
  const [isLoginView, setIsLoginView] = useState(true);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [formData, setFormData] = useState({ email: '', password: '', name: '', role: 'student' });

  const handleChange = (e) => setFormData({ ...formData, [e.target.name]: e.target.value });

  const handleSubmit = async (e) => {
    e.preventDefault(); setLoading(true); setError('');
    const API_AUTH = '/api/v1/auth';
    try {
      if (isLoginView) {
        const res = await fetch(`${API_AUTH}/login`, {
          method: 'POST', headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({email: formData.email, password: formData.password})
        });
        const data = await res.json(); if (!res.ok) throw new Error(data.detail);
        onLoginSuccess(data.access_token);
      } else {
        const res = await fetch(`${API_AUTH}/register`, {
          method: 'POST', headers: {'Content-Type': 'application/json'},
          body: JSON.stringify(formData)
        });
        const data = await res.json(); if (!res.ok) throw new Error(data.detail);
        alert('Registered!'); setIsLoginView(true);
      }
    } catch (err) { setError(err.message); } finally { setLoading(false); }
  };

  return (
    <div className="auth-container"><div className="card auth-card">
      <h2>{isLoginView ? 'Login' : 'Register'}</h2>{error && <div className="error-msg">{error}</div>}
      <form onSubmit={handleSubmit}>
        <div className="form-group"><label>Email</label><input name="email" type="email" required value={formData.email} onChange={handleChange} /></div>
        <div className="form-group"><label>Password</label><input name="password" type="password" required value={formData.password} onChange={handleChange} /></div>
        {!isLoginView && (<><div className="form-group"><label>Name</label><input name="name" value={formData.name} onChange={handleChange} /></div><div className="form-group"><label>Role</label><select name="role" value={formData.role} onChange={handleChange}><option value="student">Student</option><option value="teacher">Teacher</option></select></div></>)}
        <button type="submit" className="btn-submit" disabled={loading}>{loading ? '...' : (isLoginView ? 'Login' : 'Register')}</button>
      </form>
      <p className="toggle-auth"><span onClick={()=>setIsLoginView(!isLoginView)}>{isLoginView?"Create account":"Back to Login"}</span></p>
    </div></div>
  );
};

const UploadCard = ({ questions, onSubmit }) => {
  const [qId, setQId] = useState(''); const [txt, setTxt] = useState(''); const [sub, setSub] = useState(false);
  const hSub = async (e) => { e.preventDefault(); if(!qId||!txt)return; setSub(true); await onSubmit({question_id:parseInt(qId), answer_text:txt}); setTxt(''); setSub(false); };
  return (
    <section className="card"><h2>Submit Homework</h2><form onSubmit={hSub}>
      <div className="form-group"><label>Question</label><select value={qId} onChange={(e)=>setQId(e.target.value)} required><option value="" disabled>Select...</option>{questions.map(q=><option key={q.id} value={q.id}>{q.title}</option>)}</select></div>
      <div className="form-group"><label>Answer</label><textarea rows="5" value={txt} onChange={(e)=>setTxt(e.target.value)} required className="input-area"/></div>
      <button className="btn-submit" disabled={sub}>{sub?'...':'Submit'}</button>
    </form></section>
  );
};

const HistoryList = ({ submissions, questions, loading }) => {
  const getQ = (id) => { const q = questions.find(i=>i.id===id); return q?q.title:`Q#${id}`; };
  return (
    <section className="card"><h2>History</h2>{loading?<p>Loading...</p>:submissions.length===0?<p>No data</p>:
      <table className="history-table"><thead><tr><th>Question</th><th>Time</th><th>Status</th><th>Score</th></tr></thead><tbody>
        {submissions.map(i=><tr key={i.id}><td><b>{getQ(i.question_id)}</b></td><td>{new Date(i.created_at).toLocaleString()}</td><td><StatusBadge status={i.status}/></td><td>{i.final_score??'--'}</td></tr>)}
      </tbody></table>}
    </section>
  );
};

const StatusBadge = ({ status }) => {
  const c = {pending_ml:{bg:'#fff7e6',color:'#fa8c16',l:'Processing'},ml_scored:{bg:'#e6f7ff',color:'#1890ff',l:'ML Scored'},graded:{bg:'#f6ffed',color:'#52c41a',l:'Graded'}}[status]||{bg:'#eee',color:'#000',l:status};
  return <span style={{padding:'4px 8px',borderRadius:'4px',background:c.bg,color:c.color,border:`1px solid ${c.color}`}}>{c.l}</span>;
};

// --- 3. 主要逻辑组件 (Main App Logic) ---
// --- 3. 主要逻辑组件 (已修复：区分老师和学生主页) ---
function AppContent() {
  const [token, setToken] = useState(localStorage.getItem('access_token'));
  const [user, setUser] = useState(null);
  const [questions, setQuestions] = useState([]);
  const [submissions, setSubmissions] = useState([]);
  const [loading, setLoading] = useState(false);

  const navigate = useNavigate();
  const API_BASE = '/api/v1';

  // 辅助变量：判断是否为老师 (兼容大小写)
  const isTeacher = user?.role?.toLowerCase() === 'teacher';

  const handleLogout = () => {
    localStorage.removeItem('access_token');
    setToken(null);
    setUser(null);
    navigate('/');
  };

  const authFetch = useCallback(async (url, options = {}) => {
    const headers = { 'Content-Type': 'application/json', ...options.headers, 'Authorization': `Bearer ${token}` };
    const response = await fetch(`${API_BASE}${url}`, { ...options, headers });
    if (response.status === 401) {
      handleLogout();
      throw new Error("Unauthorized");
    }
    return response;
  }, [token]);

  const fetchData = useCallback(async () => {
    if (!token) return;
    setLoading(true);
    try {
      const userRes = await authFetch('/users/me');
      if (userRes.ok) setUser(await userRes.json());

      const qRes = await authFetch('/questions/');
      if (qRes.ok) setQuestions(await qRes.json());

      // 只有学生才需要加载自己的提交记录，老师可能需要别的（这里暂时保持简单）
      const subRes = await authFetch('/submissions/me');
      if (subRes.ok) setSubmissions(await subRes.json());
    } catch (err) { console.error("Fetch error:", err); }
    finally { setLoading(false); }
  }, [token, authFetch]);

  useEffect(() => { if (token) fetchData(); }, [token, fetchData]);

  const handleLoginSuccess = (accessToken) => {
    localStorage.setItem('access_token', accessToken);
    setToken(accessToken);
  };

  const handleSubmitHomework = async (payload) => {
    try {
      const res = await authFetch('/submissions/', { method: 'POST', body: JSON.stringify(payload) });
      if (!res.ok) { const err = await res.json(); alert(`Failed: ${err.detail}`); return; }
      alert('Success!');
      const subRes = await authFetch('/submissions/me');
      if (subRes.ok) setSubmissions(await subRes.json());
    } catch (err) { alert('Error submitting.'); }
  };

  if (!token) return <AuthPage onLoginSuccess={handleLoginSuccess} />;

  return (
    <div className="container">
      {/* 传递 isTeacher 给 Header，方便显示/隐藏按钮 */}
      <Header user={user} onLogout={handleLogout} isTeacher={isTeacher} />

      <Routes>
        {/* --- 首页路由：核心修改 --- */}
        <Route path="/" element={
          isTeacher ? (
            // [老师视图] 显示控制面板
            <div className="teacher-dashboard" style={{maxWidth: '800px', margin: '40px auto', textAlign: 'center'}}>
              <div className="card">
                <h2>👩‍🏫 Teacher Dashboard</h2>
                <p>Welcome back, {user?.name}.</p>
                <div style={{marginTop: '20px'}}>
                  <p>You can upload new homework questions here:</p>
                  <button
                    className="btn-submit"
                    style={{fontSize: '1.2rem', padding: '15px 30px'}}
                    onClick={() => navigate('/create-question')}
                  >
                    + Create New Question
                  </button>
                </div>
              </div>

              {/* 这里可以加一个简单的题目列表预览 */}
              <div className="card" style={{marginTop: '20px', textAlign: 'left'}}>
                <h3>Existing Questions</h3>
                <ul>
                  {questions.map(q => (
                    <li key={q.id} style={{padding:'8px 0', borderBottom:'1px solid #eee'}}>
                      #{q.id}: <b>{q.title}</b> (Max Score: {q.max_score})
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          ) : (
            // [学生视图] 保持原样
            <div className="main-grid">
              <UploadCard questions={questions} onSubmit={handleSubmitHomework} />
              <HistoryList submissions={submissions} questions={questions} loading={loading} />
            </div>
          )
        } />

        {/* 创建题目页 */}
        <Route path="/create-question" element={
          <div style={{ maxWidth: '800px', margin: '0 auto' }}>
            <button onClick={() => navigate('/')} className="btn-back">← Back to Dashboard</button>
            <CreateQuestion />
          </div>
        } />
      </Routes>
    </div>
  );
}

// --- 5. 根组件 (包裹 Router) ---
export default function App() {
  return (
    <Router>
      <AppContent />
    </Router>
  );
}