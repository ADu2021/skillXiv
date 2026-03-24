import { useState } from 'react'
import { useNavigate } from 'react-router-dom'

export default function SearchBar({ large = false, initialQuery = '' }) {
  const [query, setQuery] = useState(initialQuery)
  const navigate = useNavigate()

  const handleSearch = (e) => {
    e.preventDefault()
    if (query.trim()) {
      navigate(`/browse?q=${encodeURIComponent(query.trim())}`)
    } else {
      navigate('/browse')
    }
  }

  return (
    <form className={`search-bar ${large ? 'search-bar-large' : ''}`} onSubmit={handleSearch}>
      <svg className="search-bar-icon" width={large ? 22 : 18} height={large ? 22 : 18} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <circle cx="11" cy="11" r="8"/>
        <path d="m21 21-4.3-4.3"/>
      </svg>
      <input
        type="text"
        placeholder="Search skills by name, description, or paper title..."
        value={query}
        onChange={e => setQuery(e.target.value)}
      />
      <button type="submit" className="search-bar-btn">Search</button>
    </form>
  )
}
