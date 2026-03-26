import { useState, useEffect, useMemo } from 'react'
import { useSearchParams } from 'react-router-dom'
import SearchBar from '../components/SearchBar'
import SkillCard from '../components/SkillCard'

const PAGE_SIZE = 24

export default function BrowsePage() {
  const [skills, setSkills] = useState([])
  const [sourcesMeta, setSourcesMeta] = useState([])
  const [selectedSources, setSelectedSources] = useState(null) // null = all selected
  const [loading, setLoading] = useState(true)
  const [searchParams] = useSearchParams()
  const query = searchParams.get('q') || ''
  const [page, setPage] = useState(1)

  useEffect(() => {
    Promise.all([
      fetch('./skills-index.json').then(r => r.json()),
      fetch('./sources-meta.json').then(r => r.json()).catch(() => [])
    ]).then(([skillsData, sourcesData]) => {
      setSkills(skillsData)
      setSourcesMeta(sourcesData)
      // Default: all sources selected
      setSelectedSources(new Set(sourcesData.map(s => s.id)))
      setLoading(false)
    }).catch(() => setLoading(false))
  }, [])

  // Reset page when query or sources change
  useEffect(() => { setPage(1) }, [query, selectedSources])

  const toggleSource = (sourceId) => {
    setSelectedSources(prev => {
      // If all are currently selected, narrow down to just the clicked one
      if (prev.size === sourcesMeta.length) {
        return new Set([sourceId])
      }
      // If only one is selected, switch to the clicked one (or back to all if same)
      if (prev.size === 1) {
        if (prev.has(sourceId)) {
          return new Set(sourcesMeta.map(s => s.id))
        }
        return new Set([sourceId])
      }
      // Multi-select: toggle individual sources, keep at least one
      const next = new Set(prev)
      if (next.has(sourceId)) {
        if (next.size > 1) next.delete(sourceId)
      } else {
        next.add(sourceId)
      }
      return next
    })
  }

  const selectAll = () => {
    setSelectedSources(new Set(sourcesMeta.map(s => s.id)))
  }

  const allSelected = selectedSources && sourcesMeta.length > 0 &&
    selectedSources.size === sourcesMeta.length

  const filtered = useMemo(() => {
    let result = skills

    // Filter by selected sources
    if (selectedSources && selectedSources.size < sourcesMeta.length) {
      result = result.filter(s => selectedSources.has(s.source))
    }

    // Filter by search query
    if (query) {
      const q = query.toLowerCase()
      const terms = q.split(/\s+/)
      result = result.filter(s => {
        const text = `${s.name} ${s.description} ${s.paperTitle} ${s.keywords || ''}`.toLowerCase()
        return terms.every(t => text.includes(t))
      })
    }

    return result
  }, [skills, query, selectedSources, sourcesMeta.length])

  const totalPages = Math.ceil(filtered.length / PAGE_SIZE)
  const visible = filtered.slice(0, page * PAGE_SIZE)

  // Count skills per source for badges
  const sourceCounts = useMemo(() => {
    const counts = {}
    for (const s of skills) {
      counts[s.source] = (counts[s.source] || 0) + 1
    }
    return counts
  }, [skills])

  return (
    <div className="browse">
      <div className="section-inner">
        <div className="browse-header">
          <h1 className="browse-title">Browse Skills</h1>
          <p className="browse-count">
            {query
              ? `${filtered.length} result${filtered.length !== 1 ? 's' : ''} for "${query}"`
              : `${filtered.length} skills available`}
          </p>
        </div>
        <SearchBar initialQuery={query} />

        {/* Source filter */}
        {sourcesMeta.length > 1 && (
          <div className="source-filter">
            <div className="source-filter-label">Sources</div>
            <div className="source-filter-chips">
              <button
                className={`source-chip ${allSelected ? 'source-chip-active' : ''}`}
                onClick={selectAll}
              >
                All
              </button>
              {sourcesMeta.map(source => {
                const active = selectedSources && selectedSources.has(source.id)
                return (
                  <button
                    key={source.id}
                    className={`source-chip ${active && !allSelected ? 'source-chip-active' : ''}`}
                    onClick={() => toggleSource(source.id)}
                    title={source.id}
                  >
                    {source.label}
                    <span className="source-chip-count">{sourceCounts[source.id] || 0}</span>
                  </button>
                )
              })}
            </div>
          </div>
        )}

        {loading ? (
          <div className="loading">Loading skills...</div>
        ) : filtered.length === 0 ? (
          <div className="no-results">
            <h3>No skills found</h3>
            <p>Try adjusting your search terms or source filters.</p>
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
