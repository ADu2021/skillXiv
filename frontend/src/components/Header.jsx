import { Link, useNavigate } from 'react-router-dom'
import { useState } from 'react'
import useSkillCount from '../hooks/useSkillCount'

export default function Header() {
  const [query, setQuery] = useState('')
  const navigate = useNavigate()
  const skillCount = useSkillCount()

  const handleSearch = (e) => {
    e.preventDefault()
    if (query.trim()) {
      navigate(`/browse?q=${encodeURIComponent(query.trim())}`)
    }
  }

  return (
    <header className="header">
      <div className="header-inner">
        <Link to="/" className="logo">
          <span className="logo-icon">S</span>
          <span className="logo-text">Skill<span className="logo-highlight">Xiv</span></span>
        </Link>
        <form className="header-search" onSubmit={handleSearch}>
          <svg className="search-icon" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="11" cy="11" r="8"/>
            <path d="m21 21-4.3-4.3"/>
          </svg>
          <input
            type="text"
            placeholder={skillCount ? `Search ${skillCount.toLocaleString()}+ skills...` : 'Search skills...'}
            value={query}
            onChange={e => setQuery(e.target.value)}
          />
        </form>
        <nav className="header-nav">
          <Link to="/browse" className="nav-link">Browse</Link>
          <a href="https://arxiv.org" target="_blank" rel="noopener noreferrer" className="nav-link">arXiv</a>
        </nav>
      </div>
    </header>
  )
}
