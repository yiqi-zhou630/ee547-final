import React, { useState, useEffect, useCallback } from 'react';
import { BrowserRouter as Router, Routes, Route, useNavigate } from 'react-router-dom';
import './App.css';
import CreateQuestion from './CreateQuestion';
import TeacherGradingPanel from './TeacherGradingPanel';

// ================= Header =================
const Header = ({ user, onLogout }) => {
  const navigate = useNavigate();
  const isTeacher = user?.role?.toLowerCase() === 'teacher';

  return (
    <header className="header">
      <h1>EE547 Grading System</h1>
      <div className="header-right">
        {user ? (
          <span className="user-info">
            {user.name} ({user.role})
          </span>
        ) : (
          <span className="user-info">...</span>
        )}

        {isTeacher && (
          <button
            className="btn-nav"
            onClick={() => navigate('/create-question')}
          >
            + New Question
          </button>
        )}

        <button onClick={onLogout} className="btn-logout">
          Logout
        </button>
      </div>
    </header>
  );
};

// ================= AuthPage =================
const AuthPage = ({ onLoginSuccess }) => {
  const [isLoginView, setIsLoginView] = useState(true);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [formData, setFormData] = useState({
    email: '',
    password: '',
    name: '',
    role: 'student',
  });

  const handleChange = (e) =>
    setFormData({ ...formData, [e.target.name]: e.target.value });

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    const API_AUTH = '/api/v1/auth';

    try {
      if (isLoginView) {
        // 登录拿 token
        const res = await fetch(`${API_AUTH}/login`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            email: formData.email,
            password: formData.password,
          }),
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || 'Login failed');

        const accessToken = data.access_token;

        // 用 token 拉取用户信息（包含 role）
        const meRes = await fetch('/api/v1/users/me', {
          headers: {
            'Content-Type': 'application/json',
            Authorization: `Bearer ${accessToken}`,
          },
        });
        const userData = await meRes.json();
        if (!meRes.ok) throw new Error(userData.detail || 'Failed to load user info');

        onLoginSuccess(accessToken, userData);
      } else {
        // 注册
        const res = await fetch(`${API_AUTH}/register`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(formData),
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || 'Register failed');
        alert('Registered! Please login.');
        setIsLoginView(true);
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="auth-container">
      <div className="card auth-card">
        <h2>{isLoginView ? 'Login' : 'Register'}</h2>
        {error && <div className="error-msg">{error}</div>}

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

          {!isLoginView && (
            <>
              <div className="form-group">
                <label>Name</label>
                <input
                  name="name"
                  value={formData.name}
                  onChange={handleChange}
                />
              </div>
              <div className="form-group">
                <label>Role</label>
                <select
                  name="role"
                  value={formData.role}
                  onChange={handleChange}
                >
                  <option value="student">Student</option>
                  <option value="teacher">Teacher</option>
                </select>
              </div>
            </>
          )}

          <button type="submit" className="btn-submit" disabled={loading}>
            {loading ? '...' : isLoginView ? 'Login' : 'Register'}
          </button>
        </form>

        <p className="toggle-auth">
          <span onClick={() => setIsLoginView(!isLoginView)}>
            {isLoginView ? 'Create account' : 'Back to Login'}
          </span>
        </p>
      </div>
    </div>
  );
};

// ================= 学生提交相关组件 =================
const UploadCard = ({ questions, onSubmit }) => {
  const [qId, setQId] = useState('');
  const [txt, setTxt] = useState('');
  const [sub, setSub] = useState(false);

  const selectedQuestion = questions.find(q => q.id === Number(qId));

  const hSub = async (e) => {
    e.preventDefault();
    if (!qId || !txt) return;
    setSub(true);
    await onSubmit({ question_id: parseInt(qId), answer_text: txt });
    setTxt('');
    setSub(false);
  };

  return (
    <section className="card">
      <h2>Submit Homework</h2>
      <form onSubmit={hSub}>
        <div className="form-group">
          <label>Question</label>
          <select
            value={qId}
            onChange={(e) => setQId(e.target.value)}
            required
          >
            <option value="" disabled>
              Select...
            </option>
            {questions.map((q) => (
              <option key={q.id} value={q.id}>
                {q.title}
              </option>
            ))}
          </select>
        </div>

        {selectedQuestion && (
          <div
            className="question-preview"
            style={{
              background: '#fafafa',
              border: '1px solid #eee',
              borderRadius: 4,
              padding: '8px 10px',
              marginBottom: 12,
              fontSize: '0.9rem',
              textAlign: 'left',
              whiteSpace: 'pre-wrap',
            }}
          >
            <div style={{ fontWeight: 'bold', marginBottom: 4 }}>
              {selectedQuestion.title}
            </div>
            <div>{selectedQuestion.question_text}</div>
          </div>
        )}

        <div className="form-group">
          <label>Answer</label>
          <textarea
            rows="5"
            value={txt}
            onChange={(e) => setTxt(e.target.value)}
            required
            className="input-area"
          />
        </div>

        <button className="btn-submit" disabled={sub}>
          {sub ? '...' : 'Submit'}
        </button>
      </form>
    </section>
  );
};


const StatusBadge = ({ status }) => {
  const c =
    {
      pending_ml: { bg: '#fff7e6', color: '#fa8c16', l: 'Processing' },
      ml_scored: { bg: '#e6f7ff', color: '#1890ff', l: 'ML Scored' },
      graded: { bg: '#f6ffed', color: '#52c41a', l: 'Graded' },
    }[status] || { bg: '#eee', color: '#000', l: status };
  return (
    <span
      style={{
        padding: '4px 8px',
        borderRadius: '4px',
        background: c.bg,
        color: c.color,
        border: `1px solid ${c.color}`,
      }}
    >
      {c.l}
    </span>
  );
};

const HistoryList = ({ submissions, questions, loading }) => {
  const getQ = (id) => {
    const q = questions.find((i) => i.id === id);
    return q ? q.title : `Q#${id}`;
  };
  return (
    <section className="card">
      <h2>History</h2>
      {loading ? (
        <p>Loading...</p>
      ) : submissions.length === 0 ? (
        <p>No data</p>
      ) : (
        <table className="history-table">
          <thead>
            <tr>
              <th>Question</th>
              <th>Time</th>
              <th>Status</th>
              <th>Score</th>
            </tr>
          </thead>
          <tbody>
            {submissions.map((i) => (
              <tr key={i.id}>
                <td>
                  <b>{getQ(i.question_id)}</b>
                </td>
                <td>{new Date(i.created_at).toLocaleString()}</td>
                <td>
                  <StatusBadge status={i.status} />
                </td>
                <td>{i.final_score ?? '--'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </section>
  );
};

// ================= 主逻辑组件 =================
function AppContent() {
  const [token, setToken] = useState(localStorage.getItem('access_token'));
  const [user, setUser] = useState(null);
  const [questions, setQuestions] = useState([]);
  const [submissions, setSubmissions] = useState([]);
  const [pendingScores, setPendingScores] = useState([]);
  const [loading, setLoading] = useState(true);
  const [gradingLoading, setGradingLoading] = useState(false);

  const navigate = useNavigate();
  const API_BASE = '/api/v1';
  const isTeacher = user?.role?.toLowerCase() === 'teacher';

  const handleLogout = () => {
    localStorage.removeItem('access_token');
    localStorage.removeItem('user_role');
    setToken(null);
    setUser(null);
    navigate('/');
  };

  const authFetch = useCallback(
    async (url, options = {}) => {
      const headers = {
        'Content-Type': 'application/json',
        ...options.headers,
        Authorization: `Bearer ${token}`,
      };
      const response = await fetch(`${API_BASE}${url}`, { ...options, headers });
      if (response.status === 401) {
        handleLogout();
        throw new Error('Unauthorized');
      }
      return response;
    },
    [token]
  );

  // 拉取用户信息 + 题目 + 学生提交 + 老师待评分列表
  const fetchData = useCallback(async () => {
    if (!token) {
      setLoading(false);
      return;
    }

    setLoading(true);
    try {
      // 1. 用户信息
      const userRes = await authFetch('/users/me');
      let userData = null;
      if (userRes.ok) {
        userData = await userRes.json();
        setUser(userData);
        if (userData.role) {
          localStorage.setItem('user_role', userData.role.toLowerCase());
        }
      }

      // 2. 所有题目
      const qRes = await authFetch('/questions/');
      if (qRes.ok) setQuestions(await qRes.json());

      // 3. 学生自己的提交
      try {
        const subRes = await authFetch('/submissions/me');
        if (subRes.ok) setSubmissions(await subRes.json());
      } catch {
        console.log('No submissions fetched (maybe teacher)');
      }

      // 4. 老师待人工评分列表
      const role = userData?.role?.toLowerCase();
      if (role === 'teacher') {
        try {
          const pendingRes = await authFetch('/scores/pending');
          if (pendingRes.ok) {
            const pendingData = await pendingRes.json();
            setPendingScores(pendingData);
          }
        } catch (e) {
          console.error('Failed to fetch pending scores', e);
        }
      } else {
        setPendingScores([]);
      }
    } catch (err) {
      console.error('Fetch error:', err);
    } finally {
      setLoading(false);
    }
  }, [token, authFetch]);

  useEffect(() => {
    if (token) {
      fetchData();
    } else {
      setLoading(false);
    }
  }, [token, fetchData]);

  // 登录成功：拿到 token + user
  const handleLoginSuccess = (accessToken, userData) => {
    localStorage.setItem('access_token', accessToken);
    if (userData.role) {
      localStorage.setItem('user_role', userData.role.toLowerCase());
    }
    setToken(accessToken);
    setUser(userData);
    // fetchData 会在 useEffect 里跑
  };

  // 学生提交作业
  const handleSubmitHomework = async (payload) => {
    try {
      const res = await authFetch('/submissions/', {
        method: 'POST',
        body: JSON.stringify(payload),
      });
      if (!res.ok) {
        const err = await res.json();
        alert(`Failed: ${err.detail}`);
        return;
      }
      alert('Success!');
      const subRes = await authFetch('/submissions/me');
      if (subRes.ok) setSubmissions(await subRes.json());
    } catch (err) {
      alert('Error submitting.');
    }
  };

  // 老师提交最终分数
  const handleSubmitGrade = async (submissionId, finalScore, comment) => {
    try {
      setGradingLoading(true);
      const res = await authFetch(`/scores/${submissionId}`, {
        method: 'PUT',
        body: JSON.stringify({
          final_score: finalScore,
          teacher_comment: comment,
        }),
      });
      if (!res.ok) {
        const errData = await res.json();
        alert(`Failed to update score: ${errData.detail || 'Unknown error'}`);
        return;
      }
      alert('Score updated!');
      // 更新 pending 列表 / 以及学生视图中的状态
      await fetchData();
    } catch (err) {
      console.error(err);
      alert('Error updating score.');
    } finally {
      setGradingLoading(false);
    }
  };

  if (!token) return <AuthPage onLoginSuccess={handleLoginSuccess} />;

  if (loading && !user) {
    return (
      <div style={{ display: 'flex', justifyContent: 'center', marginTop: '50px' }}>
        <h2>Loading user profile...</h2>
      </div>
    );
  }

  return (
    <div className="container">
      <Header user={user} onLogout={handleLogout} />

      <Routes>
        <Route
          path="/"
          element={
            isTeacher ? (
              // 老师视图
              <div
                className="teacher-dashboard"
                style={{ maxWidth: '900px', margin: '40px auto', textAlign: 'center' }}
              >
                <div className="card">
                  <h2>👩‍🏫 Teacher Dashboard</h2>
                  <p>Welcome back, {user?.name}.</p>
                  <div style={{ marginTop: '20px' }}>
                    <p>You can upload new homework questions here:</p>
                    <button
                      className="btn-submit"
                      style={{ fontSize: '1.2rem', padding: '15px 30px' }}
                      onClick={() => navigate('/create-question')}
                    >
                      + Create New Question
                    </button>
                  </div>
                </div>

                <div className="card" style={{ marginTop: '20px', textAlign: 'left' }}>
                  <h3>Existing Questions</h3>
                  {questions.length === 0 ? (
                    <p>No questions created yet.</p>
                  ) : (
                    <ul>
                      {questions.map((q) => (
                        <li
                          key={q.id}
                          style={{ padding: '8px 0', borderBottom: '1px solid #eee' }}
                        >
                          #{q.id}: <b>{q.title}</b> (Max Score: {q.max_score})
                        </li>
                      ))}
                    </ul>
                  )}
                </div>

                <TeacherGradingPanel
                  pendingScores={pendingScores}
                  questions={questions}
                  onSubmitGrade={handleSubmitGrade}
                  loading={gradingLoading}
                />
              </div>
            ) : (
              // 学生视图
              <div className="main-grid">
                <UploadCard questions={questions} onSubmit={handleSubmitHomework} />
                <HistoryList
                  submissions={submissions}
                  questions={questions}
                  loading={loading}
                />
              </div>
            )
          }
        />

        <Route
          path="/create-question"
          element={
            <div style={{ maxWidth: '800px', margin: '0 auto' }}>
              <button
                onClick={() => navigate('/')}
                className="btn-back"
                style={{ marginBottom: '10px', display: 'block' }}
              >
                ← Back to Dashboard
              </button>
              <CreateQuestion />
            </div>
          }
        />
      </Routes>
    </div>
  );
}

// 根组件
export default function App() {
  return (
    <Router>
      <AppContent />
    </Router>
  );
}
