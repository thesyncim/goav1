// SPDX-License-Identifier: BSD-2-Clause
//
// See LICENSE for the BSD-2-Clause grant.

package threading

import "sync"

// WaitList parks goroutines on a monotonic progress value and wakes only the
// ones whose need the published value meets. A sync.Cond broadcast wakes
// every waiter on every publish, and the ones still short of their need park
// again: a wake, a reschedule and a park for nothing, several per needed
// wake when waiters on different rows share one condition. dav1d's workers
// likewise sleep on their own condition and are signalled for the work they
// wait on.
//
// A waiter registers a key (what it waits on, such as a row) and the value it
// needs; Wake(key, v) releases the waiters of key that v satisfies. The
// caller's mutex guards every method. Each slot owns its wake channel, made
// when the list first grows to it and reused afterwards, so a list that has
// reached its peak waiter count parks and wakes without allocating.
type WaitList struct {
	slots   []waitSlot
	waiting int
}

type waitSlot struct {
	ch    chan struct{}
	key   int32
	need  int32
	state uint8
}

const (
	waitSlotFree    = iota
	waitSlotWaiting // registered, not yet released
	waitSlotWoken   // released; its waiter has not yet reclaimed it
)

// Waiting reports how many registered goroutines no Wake has released yet.
func (l *WaitList) Waiting() int { return l.waiting }

// Park registers a waiter for key needing need, releases mu, blocks until a
// Wake or WakeAll releases it, and returns with mu held again, like
// sync.Cond.Wait. The caller re-checks its condition: WakeAll releases
// waiters whose need is unmet.
func (l *WaitList) Park(mu *sync.Mutex, key, need int32) {
	i := 0
	for i < len(l.slots) && l.slots[i].state != waitSlotFree {
		i++
	}
	if i == len(l.slots) {
		l.slots = append(l.slots, waitSlot{ch: make(chan struct{}, 1)})
	}
	s := &l.slots[i]
	s.key, s.need, s.state = key, need, waitSlotWaiting
	l.waiting++
	ch := s.ch
	mu.Unlock()
	<-ch
	mu.Lock()
	// The slot is free only once its wake is consumed, so a later waiter
	// never receives this one's token. The slice may have grown meanwhile.
	l.slots[i].state = waitSlotFree
}

// Wake releases the waiters of key whose need v meets.
func (l *WaitList) Wake(key, v int32) {
	if l.waiting == 0 {
		return
	}
	for i := range l.slots {
		s := &l.slots[i]
		if s.state == waitSlotWaiting && s.key == key && v >= s.need {
			l.release(s)
		}
	}
}

// WakeAll releases every waiter.
func (l *WaitList) WakeAll() {
	if l.waiting == 0 {
		return
	}
	for i := range l.slots {
		if s := &l.slots[i]; s.state == waitSlotWaiting {
			l.release(s)
		}
	}
}

func (l *WaitList) release(s *waitSlot) {
	s.state = waitSlotWoken
	l.waiting--
	// The channel holds at most this one wake: the slot is not reused
	// until its waiter received it.
	s.ch <- struct{}{}
}
