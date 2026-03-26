import { useState, useEffect } from 'react'

// Shared cache so multiple components don't re-fetch
let cachedCount = null
let fetchPromise = null

export default function useSkillCount() {
  const [count, setCount] = useState(cachedCount)

  useEffect(() => {
    if (cachedCount !== null) {
      setCount(cachedCount)
      return
    }
    if (!fetchPromise) {
      fetchPromise = fetch('./skills-index.json')
        .then(r => r.json())
        .then(data => {
          // Support both old format (array) and new format ({ skills, globalTags })
          const skills = Array.isArray(data) ? data : (data.skills || [])
          cachedCount = skills.length
          return cachedCount
        })
        .catch(() => null)
    }
    fetchPromise.then(n => {
      if (n !== null) setCount(n)
    })
  }, [])

  return count
}
