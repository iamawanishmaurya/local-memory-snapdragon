import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useState } from 'react'
import { Loader2, Palette, Pause, Play, ShieldCheck, Trash2, Zap } from 'lucide-react'
import { toast } from 'sonner'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card'
import { Separator } from '@/components/ui/separator'
import { ConfigDrawer } from '@/components/config-drawer'
import { Header } from '@/components/layout/header'
import { Main } from '@/components/layout/main'
import { ProfileDropdown } from '@/components/profile-dropdown'
import { Search } from '@/components/search'
import { ThemeSwitch } from '@/components/theme-switch'
import { SidebarNav } from './components/sidebar-nav'
import { api } from '@/lib/api'

const sidebarNavItems = [
  {
    title: 'Privacy & Data',
    href: '/settings',
    icon: <ShieldCheck size={18} />,
  },
  {
    title: 'Appearance',
    href: '/settings/appearance',
    icon: <Palette size={18} />,
  },
]

export function PrivacySettings() {
  const qc = useQueryClient()
  const [confirmText, setConfirmText] = useState('')
  const status = useQuery({ queryKey: ['status'], queryFn: api.status, refetchInterval: 5000 })
  const network = useQuery({ queryKey: ['network'], queryFn: api.network, refetchInterval: 5000 })

  const togglePause = useMutation({
    mutationFn: () => (status.data?.paused ? api.resume() : api.pause()),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['status'] }),
  })

  const wipe = useMutation({
    mutationFn: api.wipe,
    onSuccess: () => {
      toast.success('Local index wiped. Nothing remains on disk.')
      setConfirmText('')
      qc.invalidateQueries()
    },
    onError: (e) => toast.error(String(e)),
  })

  const s = status.data
  const canWipe = confirmText.toUpperCase() === 'WIPE'

  return (
    <>
      <Header>
        <Search className='me-auto' />
        <ThemeSwitch />
        <ConfigDrawer />
        <ProfileDropdown />
      </Header>

      <Main fixed>
        <div className='space-y-0.5'>
          <h1 className='text-2xl font-bold tracking-tight md:text-3xl'>Privacy & Data</h1>
          <p className='text-muted-foreground'>
            Local Memory is 100% on-device. No network calls, no telemetry, no cloud.
          </p>
        </div>
        <Separator className='my-4 lg:my-6' />
        <div className='flex flex-1 flex-col space-y-2 overflow-hidden md:space-y-2 lg:flex-row lg:space-y-0 lg:space-x-12'>
          <aside className='top-0 lg:sticky lg:w-1/5'>
            <SidebarNav items={sidebarNavItems} />
          </aside>
          <div className='w-full space-y-4 overflow-y-auto p-1'>
            <Card>
              <CardHeader>
                <CardTitle className='flex items-center gap-2 text-base'>
                  <ShieldCheck className='size-4 text-green-600' /> Privacy status
                </CardTitle>
                <CardDescription>
                  The server binds to 127.0.0.1 only and makes zero outbound connections.
                </CardDescription>
              </CardHeader>
              <CardContent className='flex flex-wrap gap-2'>
                <Badge variant='secondary'>localhost only</Badge>
                <Badge variant='secondary'>zero telemetry</Badge>
                <Badge variant='secondary'>offline after setup</Badge>
                <Badge variant={s?.npu.qnn_available ? 'default' : 'secondary'} className='gap-1'>
                  <Zap className='size-3' />
                  {s?.npu.qnn_available ? 'NPU: QNN active' : 'CPU fallback'}
                </Badge>
                {s && (
                  <Badge variant='secondary'>
                    {s.stats.files.toLocaleString()} files · {s.stats.chunks.toLocaleString()} chunks
                  </Badge>
                )}
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className='flex items-center gap-2 text-base'>
                  <ShieldCheck className='size-4 text-green-600' /> Outbound network proof
                </CardTitle>
                <CardDescription>
                  Live count of connection attempts this server process made to any
                  non-local address, plus an OS-level audit of established
                  connections. Enable airplane mode and watch it stay at zero.
                </CardDescription>
              </CardHeader>
              <CardContent className='space-y-2'>
                <div className='text-3xl font-bold tabular-nums'>
                  Outbound network calls:{' '}
                  <span className={network.data && network.data.outbound_calls === 0 ? 'text-green-600' : 'text-destructive'}>
                    {network.data?.outbound_calls ?? '…'}
                  </span>{' '}
                  <span className='text-sm font-normal text-muted-foreground'>since start</span>
                </div>
                <p className='text-xs text-muted-foreground'>
                  {network.data?.since && <>monitoring since {new Date(network.data.since).toLocaleTimeString()} · </>}
                  last audit: {network.data?.established_non_loopback ?? 0} established
                  non-local connection(s)
                  {network.data?.last_audit && <> (checked {new Date(network.data.last_audit).toLocaleTimeString()})</>}
                </p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className='text-base'>Indexing</CardTitle>
                <CardDescription>
                  Pause the watcher and indexing pipeline at any time. Search keeps working on what
                  is already indexed.
                </CardDescription>
              </CardHeader>
              <CardContent>
                <Button onClick={() => togglePause.mutate()} disabled={togglePause.isPending}>
                  {togglePause.isPending ? (
                    <Loader2 className='animate-spin' />
                  ) : s?.paused ? (
                    <Play />
                  ) : (
                    <Pause />
                  )}
                  {s?.paused ? 'Resume indexing' : 'Pause indexing'}
                </Button>
              </CardContent>
            </Card>

            <Card className='border-destructive/50'>
              <CardHeader>
                <CardTitle className='text-base text-destructive'>Full local wipe</CardTitle>
                <CardDescription>
                  Permanently deletes the entire index: database, embeddings, thumbnails and
                  settings. Your actual files are never touched. This cannot be undone.
                </CardDescription>
              </CardHeader>
              <CardContent className='space-y-3'>
                <input
                  className='w-full rounded-md border bg-transparent px-3 py-2 text-sm'
                  placeholder='Type WIPE to confirm'
                  value={confirmText}
                  onChange={(e) => setConfirmText(e.target.value)}
                />
                <Button
                  variant='destructive'
                  disabled={!canWipe || wipe.isPending}
                  onClick={() => wipe.mutate()}
                >
                  {wipe.isPending ? <Loader2 className='animate-spin' /> : <Trash2 />}
                  Wipe all data
                </Button>
              </CardContent>
            </Card>
          </div>
        </div>
      </Main>
    </>
  )
}
