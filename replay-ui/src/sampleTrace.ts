export type HeldObject = {
  name: string
  position: number[]
  ingredients?: string[]
  is_cooking?: boolean
  is_ready?: boolean
}

export type PlayerFrame = {
  id: number
  name: string
  position: number[]
  orientation: number[]
  held_object: HeldObject | null
  goal: string
  goal_label: string
}

export type PotFrame = {
  position: number[]
  stage: string
  ingredients: string[]
  ingredient_count: number
}

export type ReplayFrame = {
  tick: number
  score: number
  players: PlayerFrame[]
  pots: PotFrame[]
  counter_objects: HeldObject[]
}

export type TickDecision = {
  playerId: number
  playerName: string
  goal: string
  goalLabel: string
  previousGoal: string
  previousGoalLabel: string
  changed: boolean
  action: string
}

export type TickEvent = {
  type: string
  playerId: number
  playerName: string
  message: string
}

export type ReplayTick = {
  tick: number
  actions: string[]
  decisions: TickDecision[]
  goal_changes: TickDecision[]
  events: TickEvent[]
  scoreDelta: number
  scoreAfter: number
  headline: string
}

export type ReplayData = {
  meta: {
    layout: string
    backend: string
    openai_model: string
    local_model: string
    max_ticks: number
  }
  layout: {
    width: number
    height: number
    terrain: {
      terrain: string
      theme: string
      position: number[]
    }[][]
    stations: {
      type: string
      position: number[]
    }[]
  }
  frames: ReplayFrame[]
  tick_events: ReplayTick[]
  score: number
  soups_delivered: number
}

export const sampleReplay: ReplayData = {
  meta: {
    layout: 'cramped_room',
    backend: 'mock',
    openai_model: 'mock',
    local_model: 'mock',
    max_ticks: 6,
  },
  layout: {
    width: 5,
    height: 4,
    terrain: [
      [
        { terrain: 'X', theme: 'counter', position: [0, 0] },
        { terrain: 'X', theme: 'counter', position: [1, 0] },
        { terrain: 'P', theme: 'pot', position: [2, 0] },
        { terrain: 'X', theme: 'counter', position: [3, 0] },
        { terrain: 'X', theme: 'counter', position: [4, 0] },
      ],
      [
        { terrain: 'O', theme: 'onion', position: [0, 1] },
        { terrain: ' ', theme: 'path', position: [1, 1] },
        { terrain: ' ', theme: 'path', position: [2, 1] },
        { terrain: ' ', theme: 'path', position: [3, 1] },
        { terrain: 'O', theme: 'onion', position: [4, 1] },
      ],
      [
        { terrain: 'X', theme: 'counter', position: [0, 2] },
        { terrain: ' ', theme: 'path', position: [1, 2] },
        { terrain: ' ', theme: 'path', position: [2, 2] },
        { terrain: ' ', theme: 'path', position: [3, 2] },
        { terrain: 'X', theme: 'counter', position: [4, 2] },
      ],
      [
        { terrain: 'X', theme: 'counter', position: [0, 3] },
        { terrain: 'D', theme: 'dish', position: [1, 3] },
        { terrain: 'X', theme: 'counter', position: [2, 3] },
        { terrain: 'S', theme: 'serve', position: [3, 3] },
        { terrain: 'X', theme: 'counter', position: [4, 3] },
      ],
    ],
    stations: [
      { type: 'pot', position: [2, 0] },
      { type: 'onion', position: [0, 1] },
      { type: 'onion', position: [4, 1] },
      { type: 'dish', position: [1, 3] },
      { type: 'serve', position: [3, 3] },
    ],
  },
  frames: [
    {
      tick: 0,
      score: 0,
      players: [
        { id: 0, name: 'Alice', position: [1, 2], orientation: [0, -1], held_object: null, goal: 'get_onion', goal_label: 'Grab onion' },
        { id: 1, name: 'Bob', position: [3, 2], orientation: [0, -1], held_object: null, goal: 'get_onion', goal_label: 'Grab onion' },
      ],
      pots: [{ position: [2, 0], stage: 'empty', ingredients: [], ingredient_count: 0 }],
      counter_objects: [],
    },
    {
      tick: 1,
      score: 0,
      players: [
        { id: 0, name: 'Alice', position: [1, 1], orientation: [-1, 0], held_object: null, goal: 'get_onion', goal_label: 'Grab onion' },
        { id: 1, name: 'Bob', position: [3, 1], orientation: [1, 0], held_object: null, goal: 'get_onion', goal_label: 'Grab onion' },
      ],
      pots: [{ position: [2, 0], stage: 'empty', ingredients: [], ingredient_count: 0 }],
      counter_objects: [],
    },
    {
      tick: 2,
      score: 0,
      players: [
        { id: 0, name: 'Alice', position: [1, 1], orientation: [-1, 0], held_object: { name: 'onion', position: [1, 1] }, goal: 'place_onion', goal_label: 'Fill pot' },
        { id: 1, name: 'Bob', position: [3, 1], orientation: [1, 0], held_object: { name: 'onion', position: [3, 1] }, goal: 'place_onion', goal_label: 'Fill pot' },
      ],
      pots: [{ position: [2, 0], stage: 'empty', ingredients: [], ingredient_count: 0 }],
      counter_objects: [],
    },
    {
      tick: 3,
      score: 0,
      players: [
        { id: 0, name: 'Alice', position: [2, 1], orientation: [0, -1], held_object: { name: 'onion', position: [2, 1] }, goal: 'place_onion', goal_label: 'Fill pot' },
        { id: 1, name: 'Bob', position: [2, 2], orientation: [0, -1], held_object: { name: 'onion', position: [2, 2] }, goal: 'place_onion', goal_label: 'Fill pot' },
      ],
      pots: [{ position: [2, 0], stage: 'empty', ingredients: [], ingredient_count: 0 }],
      counter_objects: [],
    },
    {
      tick: 4,
      score: 0,
      players: [
        { id: 0, name: 'Alice', position: [2, 1], orientation: [0, -1], held_object: null, goal: 'get_onion', goal_label: 'Grab onion' },
        { id: 1, name: 'Bob', position: [2, 2], orientation: [0, -1], held_object: { name: 'onion', position: [2, 2] }, goal: 'place_onion', goal_label: 'Fill pot' },
      ],
      pots: [{ position: [2, 0], stage: 'filling', ingredients: ['onion'], ingredient_count: 1 }],
      counter_objects: [],
    },
    {
      tick: 5,
      score: 0,
      players: [
        { id: 0, name: 'Alice', position: [1, 1], orientation: [-1, 0], held_object: null, goal: 'get_onion', goal_label: 'Grab onion' },
        { id: 1, name: 'Bob', position: [2, 1], orientation: [0, -1], held_object: null, goal: 'get_onion', goal_label: 'Grab onion' },
      ],
      pots: [{ position: [2, 0], stage: 'filling', ingredients: ['onion', 'onion'], ingredient_count: 2 }],
      counter_objects: [],
    },
  ],
  tick_events: [
    { tick: 1, actions: ['up', 'up'], decisions: [], goal_changes: [], events: [], scoreDelta: 0, scoreAfter: 0, headline: 'The team keeps moving through the kitchen.' },
    {
      tick: 2,
      actions: ['interact', 'interact'],
      decisions: [
        { playerId: 0, playerName: 'Alice', goal: 'place_onion', goalLabel: 'Fill pot', previousGoal: 'get_onion', previousGoalLabel: 'Grab onion', changed: true, action: 'interact' },
        { playerId: 1, playerName: 'Bob', goal: 'place_onion', goalLabel: 'Fill pot', previousGoal: 'get_onion', previousGoalLabel: 'Grab onion', changed: true, action: 'interact' },
      ],
      goal_changes: [
        { playerId: 0, playerName: 'Alice', goal: 'place_onion', goalLabel: 'Fill pot', previousGoal: 'get_onion', previousGoalLabel: 'Grab onion', changed: true, action: 'interact' },
      ],
      events: [
        { type: 'onion_pickup', playerId: 0, playerName: 'Alice', message: 'picked up an onion' },
        { type: 'onion_pickup', playerId: 1, playerName: 'Bob', message: 'picked up an onion' },
      ],
      scoreDelta: 0,
      scoreAfter: 0,
      headline: 'Alice picked up an onion',
    },
    { tick: 3, actions: ['right', 'left'], decisions: [], goal_changes: [], events: [], scoreDelta: 0, scoreAfter: 0, headline: 'The team keeps moving through the kitchen.' },
    {
      tick: 4,
      actions: ['interact', 'stay'],
      decisions: [
        { playerId: 0, playerName: 'Alice', goal: 'get_onion', goalLabel: 'Grab onion', previousGoal: 'place_onion', previousGoalLabel: 'Fill pot', changed: true, action: 'interact' },
      ],
      goal_changes: [
        { playerId: 0, playerName: 'Alice', goal: 'get_onion', goalLabel: 'Grab onion', previousGoal: 'place_onion', previousGoalLabel: 'Fill pot', changed: true, action: 'interact' },
      ],
      events: [{ type: 'potting_onion', playerId: 0, playerName: 'Alice', message: 'added an onion to the pot' }],
      scoreDelta: 0,
      scoreAfter: 0,
      headline: 'Alice added an onion to the pot',
    },
    {
      tick: 5,
      actions: ['left', 'interact'],
      decisions: [
        { playerId: 1, playerName: 'Bob', goal: 'get_onion', goalLabel: 'Grab onion', previousGoal: 'place_onion', previousGoalLabel: 'Fill pot', changed: true, action: 'interact' },
      ],
      goal_changes: [
        { playerId: 1, playerName: 'Bob', goal: 'get_onion', goalLabel: 'Grab onion', previousGoal: 'place_onion', previousGoalLabel: 'Fill pot', changed: true, action: 'interact' },
      ],
      events: [{ type: 'potting_onion', playerId: 1, playerName: 'Bob', message: 'added an onion to the pot' }],
      scoreDelta: 0,
      scoreAfter: 0,
      headline: 'Bob added an onion to the pot',
    },
  ],
  score: 0,
  soups_delivered: 0,
}
