import { useState, useEffect, useMemo } from 'react'
import { useSearchParams } from 'react-router-dom'
import SearchBar from '../components/SearchBar'
import SkillCard from '../components/SkillCard'
import TagBar from '../components/TagBar'

const PAGE_SIZE = 24

export default function BrowsePage() {
  const [skills, setSkills] = useState([])
  const [globalTags, setGlobalTags] = useState([])
  const [sourcesMeta, setSourcesMeta] = useState([])
  const [selectedSources, setSelectedSources] = useState(null) // null = all selected
  const [loading, setLoading] = useState(true)
  const [searchParams, setSearchParams] = useSearchParams()
  const query = searchParams.get('q') || ''
  const activeTag = searchParams.get('tag') || ''
  const [page, setPage] = useState(1)

  useEffect(() => {
    Promise.all([
      fetch('./skills-index.json').then(r => r.json()),
      fetch('./sources-meta.json').then(r => r.json()).catch(() => [])
    ]).then(([skillsData, sourcesData]) => {
      // Support both old format (array) and new format ({ skills, globalTags })
      if (Array.isArray(skillsData)) {
        setSkills(skillsData)
      } else {
        setSkills(skillsData.skills || [])
        setGlobalTags(skillsData.globalTags || [])
      }
      setSourcesMeta(sourcesData)
      setSelectedSources(new Set(sourcesData.map(s => s.id)))
      setLoading(false)
    }).catch(() => setLoading(false))
  }, [])

  // Reset page when query, tag, or sources change
  useEffect(() => { setPage(1) }, [query, activeTag, selectedSources])

  const toggleSource = (sourceId) => {
    setSelectedSources(prev => {
      if (prev.size === sourcesMeta.length) {
        return new Set([sourceId])
      }
      if (prev.size === 1) {
        if (prev.has(sourceId)) {
          return new Set(sourcesMeta.map(s => s.id))
        }
        return new Set([sourceId])
      }
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
    return skills.filter(s => {
      // Source filter
      const matchesSource = (() => {
        if (!selectedSources || selectedSources.size === sourcesMeta.length) return true
        return selectedSources.has(s.source)
      })()

      // Text search: AND-logic across all terms
      const matchesQuery = (() => {
        if (!query) return true
        const terms = query.toLowerCase().split(/\s+/)
        const text = `${s.name} ${s.description} ${s.paperTitle} ${s.keywords || ''}`.toLowerCase()
        return terms.every(t => text.includes(t))
      })()

      // Tag filter: skill must have the active tag
      const matchesTag = (() => {
        if (!activeTag) return true
        if (!s.tags || !Array.isArray(s.tags)) return false
        return s.tags.some(t => t.toLowerCase() === activeTag.toLowerCase())
      })()

      return matchesSource && matchesQuery && matchesTag
    })
  }, [skills, query, activeTag, selectedSources, sourcesMeta.length])

  const totalPages = Math.ceil(filtered.length / PAGE_SIZE)
  const visible = filtered.slice(0, page * PAGE_SIZE)

  const handleTagClick = (tagSlug) => {
    const params = new URLSearchParams(searchParams)
    if (tagSlug === activeTag) {
      params.delete('tag')
    } else {
      params.set('tag', tagSlug)
    }
    setSearchParams(params)
  }

  // Count skills per source for badges
  const sourceCounts = useMemo(() => {
    const counts = {}
    for (const s of skills) {
      counts[s.source] = (counts[s.source] || 0) + 1
    }
    return counts
  }, [skills])

  // Find the display name for the active tag
  const activeTagName = activeTag
    ? (globalTags.find(t => t.slug === activeTag)?.name || activeTag)
    : ''

  const resultSummary = (() => {
    const parts = []
    if (query) parts.push(`"${query}"`)
    if (activeTagName) parts.push(activeTagName)
    if (parts.length > 0) {
      return `${filtered.length} result${filtered.length !== 1 ? 's' : ''} for ${parts.join(' in ')}`
    }
    return `${skills.length} skills available`
  })()

  return (
    <div className="browse">
      <div className="section-inner">
        <div className="browse-header">
          <h1 className="browse-title">Browse Skills</h1>
          <p className="browse-count">{resultSummary}</p>
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

        {/* Tag filter */}
        {globalTags.length > 0 && (
          <TagBar
            tags={globalTags}
            activeTag={activeTag}
            onTagClick={handleTagClick}
          />
        )}

        {loading ? (
          <div className="loading">Loading skills...</div>
        ) : filtered.length === 0 ? (
          <div className="no-results">
            <h3>No skills found</h3>
            <p>Try adjusting your search terms or filters.</p>
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
