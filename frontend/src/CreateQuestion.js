// frontend/src/CreateQuestion.js
import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';

const CreateQuestion = () => {
  const [formData, setFormData] = useState({
    title: '',
    question_text: '',       // ✅ 对应后端 question_text
    reference_answer: '',    // ✅ 对应后端 reference_answer
    max_score: 5,
    topic: '',
  });
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const userRole = localStorage.getItem('user_role');

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: name === 'max_score' ? parseInt(value) || 0 : value,
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
      const response = await fetch('/api/v1/questions/', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`,
        },
        // 这里直接发 formData，字段名和后端一一对应
        body: JSON.stringify(formData),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to create question');
      }

      alert('Question created successfully!');
      navigate('/');
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

      {userRole !== 'teacher' && (
        <div
          style={{
            background: '#fffbe6',
            padding: '10px',
            marginBottom: '10px',
            border: '1px solid #ffe58f',
          }}
        >
          ⚠️ Warning: You may not have permission to perform this action if you are
          not a teacher.
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
            placeholder="e.g., Homework 1: Short Answer"
          />
        </div>

        <div className="form-group">
          <label>Question Text</label>
          <textarea
            name="question_text"
            required
            rows="4"
            value={formData.question_text}
            onChange={handleChange}
            placeholder="Enter the problem details..."
            style={{ width: '100%', padding: '8px', marginTop: '5px' }}
          />
        </div>

        <div className="form-group">
          <label>Reference Answer</label>
          <textarea
            name="reference_answer"
            required
            rows="3"
            value={formData.reference_answer}
            onChange={handleChange}
            placeholder="Enter the reference answer..."
            style={{ width: '100%', padding: '8px', marginTop: '5px' }}
          />
        </div>

        <div className="form-group">
          <label>Topic (optional)</label>
          <input
            name="topic"
            type="text"
            value={formData.topic}
            onChange={handleChange}
            placeholder="e.g., NLP, Machine Learning"
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
