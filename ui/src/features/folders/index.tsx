import { useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { FolderOpen, Loader2, Plus, RefreshCw, Trash2 } from 'lucide-react'
import { toast } from 'sonner'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardContent } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Header } from '@/components/layout/header'
import { Main } from '@/components/layout/main'
import { ProfileDropdown } from '@/components/profile-dropdown'
import { ThemeSwitch } from '@/components/theme-switch'
import { Search as GlobalSearch } from '@/components/search'
import { ConfigDrawer } from '@/components/config-drawer'
import { api } from '@/lib/api'

export function FoldersPage() {
  const qc = useQueryClient()
  const [newPath, setNewPath] = useState('')

  const status = useQuery({ queryKey: ['status'], queryFn: api.status })
  const suggestions = useQuery({ queryKey: ['suggest-folders'], queryFn: api.suggestFolders })

  const invalidate = () => {
    qc.invalidateQueries({ queryKey: ['status'] })
    qc.invalidateQueries({ queryKey: ['suggest-folders'] })
  }

  const add = useMutation({
    mutationFn: (p: string) => api.addFolder(p),
    onSuccess: () => {
      toast.success('Folder added — indexing in the background.')
      setNewPath('')
      invalidate()
    },
    onError: (e) => toast.error(String(e)),
  })
  const remove = useMutation({
    mutationFn: (p: string) => api.removeFolder(p),
    onSuccess: () => {
      toast.success('Folder removed from the index scope.')
      invalidate()
    },
  })
  const reindex = useMutation({
    mutationFn: api.reindex,
    onSuccess: () => toast.success('Re-scan started.'),
  })

  const folders = status.data?.watched_folders ?? []

  return (
    <>
      <Header>
        <GlobalSearch className='me-auto' />
        <ThemeSwitch />
        <ConfigDrawer />
        <ProfileDropdown />
      </Header>

      <Main fixed>
        <div className='mb-1 flex items-center justify-between'>
          <h1 className='text-2xl font-bold tracking-tight md:text-3xl'>Folders</h1>
          <Button variant='outline' size='sm' onClick={() => reindex.mutate()} disabled={reindex.isPending}>
            {reindex.isPending ? <Loader2 className='animate-spin' /> : <RefreshCw />}
            Re-index now
          </Button>
        </div>
        <p className='text-muted-foreground'>
          Only folders you add here are scanned. Nothing outside them is ever read.
        </p>

        <div className='mt-4 flex gap-2'>
          <Input
            value={newPath}
            onChange={(e) => setNewPath(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && newPath.trim() && add.mutate(newPath.trim())}
            placeholder='C:\Users\you\Documents'
          />
          <Button onClick={() => newPath.trim() && add.mutate(newPath.trim())} disabled={add.isPending}>
            {add.isPending ? <Loader2 className='animate-spin' /> : <Plus />} Add folder
          </Button>
        </div>

        {suggestions.data && suggestions.data.suggestions.length > 0 && (
          <div className='mt-3 flex flex-wrap items-center gap-2'>
            <span className='text-sm text-muted-foreground'>Quick add:</span>
            {suggestions.data.suggestions.map((p) => (
              <Button
                key={p}
                variant='secondary'
                size='sm'
                onClick={() => add.mutate(p)}
                disabled={folders.includes(p)}
              >
                <FolderOpen /> {p.split(/[\\/]/).pop()}
              </Button>
            ))}
          </div>
        )}

        <div className='mt-6 space-y-2'>
          {folders.length === 0 && (
            <Card className='border-dashed'>
              <CardContent className='pt-6 text-center text-muted-foreground'>
                No folders yet — add one above.
              </CardContent>
            </Card>
          )}
          {folders.map((f) => (
            <Card key={f} className='flex items-center gap-3 p-4'>
              <FolderOpen className='size-5 flex-none text-muted-foreground' />
              <span className='min-w-0 flex-1 truncate text-sm font-medium'>{f}</span>
              <Badge variant='secondary'>watched</Badge>
              <Button
                variant='ghost'
                size='icon'
                onClick={() => remove.mutate(f)}
                disabled={remove.isPending}
                aria-label={`Remove ${f}`}
              >
                <Trash2 className='text-destructive' />
              </Button>
            </Card>
          ))}
        </div>
      </Main>
    </>
  )
}
