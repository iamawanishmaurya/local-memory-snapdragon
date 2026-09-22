import {
  BarChart3,
  FolderOpen,
  HardDrive,
  HelpCircle,
  Palette,
  Search,
  Settings,
  ShieldCheck,
  BrainCircuit,
} from 'lucide-react'
import { type SidebarData } from '../types'

export const sidebarData: SidebarData = {
  user: {
    name: 'Local User',
    email: 'on-device · private',
    avatar: '',
  },
  teams: [
    {
      name: 'Local Memory',
      logo: BrainCircuit,
      plan: 'Snapdragon NPU · offline',
    },
  ],
  navGroups: [
    {
      title: 'General',
      items: [
        {
          title: 'Search',
          url: '/',
          icon: Search,
        },
        {
          title: 'Folders',
          url: '/folders',
          icon: FolderOpen,
        },
        {
          title: 'Storage Health',
          url: '/storage',
          icon: HardDrive,
        },
        {
          title: 'Statistics',
          url: '/stats',
          icon: BarChart3,
        },
      ],
    },
    {
      title: 'Privacy',
      items: [
        {
          title: 'Privacy & Data',
          url: '/settings',
          icon: ShieldCheck,
        },
      ],
    },
    {
      title: 'Other',
      items: [
        {
          title: 'Settings',
          icon: Settings,
          items: [
            {
              title: 'Privacy & Data',
              url: '/settings',
              icon: ShieldCheck,
            },
            {
              title: 'Appearance',
              url: '/settings/appearance',
              icon: Palette,
            },
          ],
        },
        {
          title: 'Help Center',
          url: '/help-center',
          icon: HelpCircle,
        },
      ],
    },
  ],
}
