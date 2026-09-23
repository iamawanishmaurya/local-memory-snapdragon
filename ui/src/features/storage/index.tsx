import { useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { AlertTriangle, FolderInput, Trash2 } from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Header } from '@/components/layout/header'
import { Main } from '@/components/layout/main'
import { ProfileDropdown } from '@/components/profile-dropdown'
import { ThemeSwitch } from '@/components/theme-switch'
import { Search as GlobalSearch } from '@/components/search'
import { ConfigDrawer } from '@/components/config-drawer'
import { api, type SuggestionFile } from '@/lib/api'

/** One-click reversible cleanup (D-03): optimistic row removal, the file
 * lands in the user-chosen cleanup folder — never deleted. */
function MoveButton({ path }: { path: string }) {
  const qc = useQueryClient()
  const move = useMutation({
    mutationFn: () => api.cleanupMove(path),
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: ['health'] })
      void qc.invalidateQueries({ queryKey: ['stats'] })
    },
  })
  return (
    <Button
      size='sm'
      variant='outline'
      className='h-7 flex-none gap-1 px-2 text-xs'
      disabled={move.isPending}
      onClick={() => move.mutate()}
    >
      {move.isPending ? (
        'Moving…'
      ) : move.isError ? (
        'Unavailable'
      ) : (
        <>
          <FolderInput className='size-3' /> Move to Cleanup
        </>
      )}
    </Button>
  )
}

function SuggestionFiles({ files }: { files: SuggestionFile[] }) {
  if (!files?.length) return null
  return (
    <div className='mt-2 space-y-1'>
      {files.map((f) => (
        <div key={f.path} className='flex items-center gap-2 text-xs'>
          <div className='min-w-0 flex-1'>
            <p className='truncate'>{f.path}</p>
            <p className='text-muted-foreground'>
              {f.size_human} · {f.reason}
            </p>
          </div>
          <MoveButton path={f.path} />
        </div>
      ))}
    </div>
  )
}

function CleanupFolderPicker({ initial }: { initial: string }) {
  const qc = useQueryClient()
  const [value, setValue] = useState(initial)
  const save = useMutation({
    mutationFn: () => api.setCleanupConfig(value),
    onSuccess: () => void qc.invalidateQueries({ queryKey: ['health'] }),
  })
  return (
    <div className='flex items-center gap-2'>
      <Input
        value={value}
        onChange={(e) => setValue(e.target.value)}
        className='h-8 text-xs'
        placeholder='Cleanup folder path'
      />
      <Button size='sm' variant='secondary' className='h-8 flex-none text-xs' disabled={save.isPending} onClick={() => save.mutate()}>
        Save
      </Button>
    </div>
  )
}

export function StorageHealthPage() {
  const health = useQuery({ queryKey: ['health'], queryFn: api.health, refetchInterval: 10_000 })
  const h = health.data

  return (
    <>
      <Header>
        <GlobalSearch className='me-auto' />
        <ThemeSwitch />
        <ConfigDrawer />
        <ProfileDropdown />
      </Header>

      <Main fixed>
        <h1 className='text-2xl font-bold tracking-tight md:text-3xl'>Storage Health</h1>
        <p className='text-muted-foreground'>
          Insights over the folders Local Memory indexes — nothing is deleted automatically.
        </p>

        {!h ? (
          <p className='mt-8 text-center text-muted-foreground'>Loading…</p>
        ) : (
          <>
            <div className='mt-4 grid gap-4 sm:grid-cols-2 lg:grid-cols-4'>
              <Card>
                <CardHeader className='pb-2'>
                  <CardTitle className='text-sm font-medium text-muted-foreground'>Total indexed</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className='text-2xl font-bold'>{h.summary.total_human}</div>
                  <p className='text-xs text-muted-foreground'>{h.summary.files.toLocaleString()} files</p>
                </CardContent>
              </Card>
              <Card>
                <CardHeader className='pb-2'>
                  <CardTitle className='text-sm font-medium text-muted-foreground'>Images</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className='text-2xl font-bold'>{h.summary.images.toLocaleString()}</div>
                  <p className='text-xs text-muted-foreground'>OCR + vision indexed</p>
                </CardContent>
              </Card>
              <Card>
                <CardHeader className='pb-2'>
                  <CardTitle className='text-sm font-medium text-muted-foreground'>Stale (&gt; 1 year)</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className='text-2xl font-bold'>{h.stale_count.toLocaleString()}</div>
                  <p className='text-xs text-muted-foreground'>candidates for archiving</p>
                </CardContent>
              </Card>
              <Card>
                <CardHeader className='pb-2'>
                  <CardTitle className='text-sm font-medium text-muted-foreground'>Duplicate groups</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className='text-2xl font-bold'>{h.dupe_groups.toLocaleString()}</div>
                  <p className='text-xs text-muted-foreground'>byte-identical copies (content hash)</p>
                </CardContent>
              </Card>
            </div>

            <div className='mt-4 grid gap-4 lg:grid-cols-2'>
              <Card>
                <CardHeader>
                  <CardTitle className='text-base'>By folder</CardTitle>
                </CardHeader>
                <CardContent className='space-y-2'>
                  {h.per_folder.map((f) => (
                    <div key={f.folder} className='flex items-center justify-between gap-2 text-sm'>
                      <span className='min-w-0 truncate'>{f.folder}</span>
                      <Badge variant='secondary'>
                        {f.count} files · {f.bytes_human}
                      </Badge>
                    </div>
                  ))}
                  {h.per_folder.length === 0 && (
                    <p className='text-muted-foreground'>No folders indexed yet.</p>
                  )}
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle className='text-base'>Cleanup suggestions</CardTitle>
                </CardHeader>
                <CardContent className='space-y-3'>
                  {h.suggestions.map((s) => (
                    <div key={s.title} className='rounded-lg border-l-4 border-amber-500 bg-amber-500/5 p-3'>
                      <p className='flex items-center gap-2 text-sm font-medium'>
                        <AlertTriangle className='size-4 text-amber-500' />
                        {s.title} — up to {s.potential_human}
                      </p>
                      <p className='mt-1 text-xs text-muted-foreground'>{s.detail}</p>
                      <SuggestionFiles files={s.files ?? []} />
                    </div>
                  ))}
                  {h.suggestions.length === 0 && (
                    <p className='text-muted-foreground'>Nothing to clean. Nice. ✨</p>
                  )}
                  <p className='flex items-center gap-1 text-xs text-muted-foreground'>
                    <Trash2 className='size-3' /> Local Memory never deletes files — one-click moves them to the
                    cleanup folder (undo = move back).
                  </p>
                  <CleanupFolderPicker initial={h.cleanup_folder ?? ''} />
                </CardContent>
              </Card>
            </div>

            <Card className='mt-4'>
              <CardHeader>
                <CardTitle className='text-base'>Largest files</CardTitle>
              </CardHeader>
              <CardContent className='space-y-2'>
                {h.largest.map((f) => (
                  <div key={f.path} className='flex items-center justify-between gap-2 text-sm'>
                    <span className='min-w-0 truncate'>{f.path}</span>
                    <Badge variant='secondary'>{f.size_human}</Badge>
                  </div>
                ))}
                {h.largest.length === 0 && (
                  <p className='text-muted-foreground'>No files indexed yet.</p>
                )}
              </CardContent>
            </Card>
          </>
        )}
      </Main>
    </>
  )
}
