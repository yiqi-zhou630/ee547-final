import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';

const CreateQuestion = () => {
  const [formData, setFormData] = useState({
    title: '',
    description: '',
    max_score: 100 // 默认满分100，匹配后端的整型字段
  });
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  // 获取用户信息用于简单的权限判断（后端也会校验）
  const userRole = localStorage.getItem('user_role'); // 假设登录时存了 role，如果没有存，后端也会拦截

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: name === 'max_score' ? parseInt(value) || 0 : value
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setLoading(true);

    const token = localStorage.getItem('access_token');
    if (!token) {
      setError('You must be logged in.');
      setLoading(false);
      return;
    }

    try {
      // 这里的路径必须匹配后端 main.py / routers 定义的路径
      const response = await fetch('/api/v1/questions/', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify(formData),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to create question');
      }

      alert('Question created successfully!');
      navigate('/'); // 创建成功后跳回主页或列表页

    } catch (err) {
      console.error(err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="card" style={{ maxWidth: '600px', margin: '2rem auto' }}>
      <h2>Teacher: Create New Question</h2>

      {/* 简单的角色提示 */}
      {userRole !== 'teacher' && (
        <div style={{ background: '#fffbe6', padding: '10px', marginBottom: '10px', border: '1px solid #ffe58f' }}>
            ️ Warning: You may not have permission to perform this action if you are not a teacher.
        </div>
      )}

      {error && <div style={{ color: 'red', marginBottom: '1rem' }}>{error}</div>}

      <form onSubmit={handleSubmit}>
        <div className="form-group">
          <label>Title</label>
          <input
            name="title"
            type="text"
            required
            value={formData.title}
            onChange={handleChange}
            placeholder="e.g., Homework 1: Linear Regression"
          />
        </div>

        <div className="form-group">
          <label>Description</label>
          <textarea
            name="description"
            required
            rows="5"
            value={formData.description}
            onChange={handleChange}
            placeholder="Enter the problem details..."
            style={{ width: '100%', padding: '8px', marginTop: '5px' }}
          />
        </div>

        <div className="form-group">
          <label>Max Score</label>
          <input
            name="max_score"
            type="number"
            required
            min="1"
            value={formData.max_score}
            onChange={handleChange}
          />
        </div>

        <button type="submit" className="btn-submit" disabled={loading}>
          {loading ? 'Creating...' : 'Publish Question'}
        </button>
      </form>
    </div>
  );
};

export default CreateQuestion;