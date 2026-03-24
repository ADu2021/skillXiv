import { useState, useEffect, useMemo } from 'react'
import { useSearchParams } from 'react-router-dom'
import SearchBar from '../components/SearchBar'
import SkillCard from '../components/SkillCard'

const PAGE_SIZE = 24

export default function BrowsePage() {
  const [skills, setSkills] = useState([])
  const [loading, setLoading] = useState(true)
  const [searchParams] = useSearchParams()
  const query = searchParams.get('q') || ''
  const [page, setPage] = useState(1)

  useEffect(() => {
    fetch('./skills-index.json')
      .then(r => r.json())
      .then(data => {
        setSkills(data)
        setLoading(false)
      })
      .catch(() => setLoading(false))
  }, [])

  // Reset page when query changes
  useEffect(() => { setPage(1) }, [query])

  const filtered = useMemo(() => {
    if (!query) return skills
    const q = query.toLowerCase()
    const terms = q.split(/\s+/)
    return skills.filter(s => {
      const text = `${s.name} ${s.description} ${s.paperTitle} ${s.keywords || ''}`.toLowerCase()
      return terms.every(t => text.includes(t))
    })
  }, [skills, query])

  const totalPages = Math.ceil(filtered.length / PAGE_SIZE)
  const visible = filtered.slice(0, page * PAGE_SIZE)

  return (
    <div className="browse">
      <div className="section-inner">
        <div className="browse-header">
          <h1 className="browse-title">Browse Skills</h1>
          <p className="browse-count">
            {query
              ? `${filtered.length} result${filtered.length !== 1 ? 's' : ''} for "${query}"`
              : `${skills.length} skills available`}
          </p>
        </div>
        <SearchBar initialQuery={query} />
        {loading ? (
          <div className="loading">Loading skills...</div>
        ) : filtered.length === 0 ? (
          <div className="no-results">
            <h3>No skills found</h3>
            <p>Try adjusting your search terms or browse all skills.</p>
          </div>
        ) : (
          <>
            <div className="skills-grid">
              {visible.map(skill => (
                <SkillCard key={skill.id} skill={skill} />
              ))}
            </div>
            {visible.length < filtered.length && (
              <div className="load-more">
                <button onClick={() => setPage(p => p + 1)} className="load-more-btn">
                  Load more ({filtered.length - visible.length} remaining)
                </button>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  )
}
