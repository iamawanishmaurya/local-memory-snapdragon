import { createFileRoute } from '@tanstack/react-router'
import { StatisticsPage } from '@/features/stats'

export const Route = createFileRoute('/_authenticated/stats/')({
  component: StatisticsPage,
})
