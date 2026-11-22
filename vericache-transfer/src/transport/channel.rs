use super::*;

verus! {

broadcast use {
    vstd::imap::group_imap_lemmas,
    vstd::iset::group_iset_lemmas,
    vstd::map::group_map_lemmas,
};

pub struct Channel<T: ByteTransport, M: ChannelAttachment = ()> {
    transport: T,
    local_node: NodeId,
    peer_node: NodeId,
    #[cfg_attr(not(verus_only), allow(dead_code))]
    instance_id: Ghost<InstanceId>,
    sent_grants: Ghost<Map<GrantTag, GrantKind>>,
    received_grants: Ghost<Map<GrantTag, GrantKind>>,
    closed: bool,
    _marker: core::marker::PhantomData<M>,
}

impl<T: ByteTransport, M: ChannelAttachment> Channel<T, M> {
    pub closed spec fn local(&self) -> NodeId {
        self.local_node
    }

    pub closed spec fn peer(&self) -> NodeId {
        self.peer_node
    }

    pub closed spec fn id(&self) -> InstanceId {
        self.instance_id@
    }

    pub closed spec fn sent(&self) -> Map<GrantTag, GrantKind> {
        self.sent_grants@
    }

    pub closed spec fn received(&self) -> Map<GrantTag, GrantKind> {
        self.received_grants@
    }

    pub closed spec fn closed(&self) -> bool {
        self.closed
    }

    pub fn is_closed(&self) -> (r: bool)
        ensures
            r == self.closed(),
    {
        self.closed
    }

    pub open spec fn same_session(&self, old: Self) -> bool {
        self.local() == old.local() && self.peer() == old.peer() && self.id() == old.id()
    }

    pub open spec fn send_preserved<C>(
        before: Self,
        after: Self,
        r: Result<(), (ChannelError<T::Error>, Tracked<Option<C>>)>,
        cap: C,
    ) -> bool {
        &&& after.same_session(before)
        &&& r.is_ok() ==> !after.closed()
        &&& r matches Err((err, ret)) ==> {
            &&& after.closed()
            &&& match err {
                ChannelError::Closed => before.closed() && ret@ == Some(cap) && after == before,
                ChannelError::Transport(_) => !before.closed() && ret@ is None,
                ChannelError::InvalidFrame => false,
            }
        }
    }

    pub fn new_with_protocol(
        transport: T,
        local_node: NodeId,
        peer_node: NodeId,
        Tracked(instance): Tracked<&TransferInstance>,
    ) -> (c: Self)
        requires
            local_node != peer_node,
            M::protocol_magic() != 0,
        ensures
            c.local() == local_node && c.peer() == peer_node && c.id() == instance.id() && c.sent()
                == Map::empty() && c.received() == Map::empty() && !c.closed(),
    {
        Self {
            transport,
            local_node,
            peer_node,
            instance_id: Ghost(instance.id()),
            sent_grants: Ghost(Map::empty()),
            received_grants: Ghost(Map::empty()),
            closed: false,
            _marker: core::marker::PhantomData,
        }
    }

    fn send(&mut self, h: Header, user: &[u8], Tracked(a): Tracked<Attachment<M>>) -> (r: Result<
        (),
        (ChannelError<T::Error>, Tracked<Option<Attachment<M>>>),
    >)
        requires
            a.valid(old(self).id(), old(self).local()),
            a.header == h,
            a.user == user@,
            h.grant() ==> !old(self).sent().contains_key(h.tag),
            !h.grant() && h.op < NOTICE ==> old(self).received().contains_key(h.tag) && old(
                self,
            ).received()[h.tag] == a.kind->Some_0,
        ensures
            Self::send_preserved(*old(self), *final(self), r, a),
            final(self).sent() == (if r.is_ok() && h.grant() {
                old(self).sent().insert(h.tag, a.kind->Some_0)
            } else {
                old(self).sent()
            }),
            final(self).received() == (if r.is_ok() && !h.grant() && h.op < NOTICE {
                old(self).received().remove(h.tag)
            } else {
                old(self).received()
            }),
    {
        if self.closed {
            return Err((ChannelError::Closed, Tracked(Some(a))));
        }
        let frame = encode_with_magic(M::magic_exec(), h, user);
        if let Err(e) = ffi::send::<T, M>(
            &mut self.transport,
            &frame,
            Tracked(a),
            Ghost(self.instance_id@),
            self.local_node,
            self.peer_node,
        ) {
            self.closed = true;
            return Err((ChannelError::Transport(e), Tracked(None)));
        }
        proof {
            if h.grant() {
                self.sent_grants = Ghost(self.sent().insert(h.tag, a.kind->Some_0));
            } else if h.op < NOTICE {
                self.received_grants = Ghost(self.received().remove(h.tag));
            }
        }
        Ok(())
    }

    fn send_with_share_and_extra(
        &mut self,
        h: Header,
        user: &[u8],
        Tracked(a): Tracked<Attachment<M>>,
    ) -> (r: Result<(), (ChannelError<T::Error>, Tracked<Option<(ShareCap, M)>>)>)
        requires
            a.valid(old(self).id(), old(self).local()),
            a.header == h,
            a.user == user@,
            a.share is Some,
            a.extra is Some,
            h.grant() ==> !old(self).sent().contains_key(h.tag),
            !h.grant() && h.op < NOTICE ==> old(self).received().contains_key(h.tag) && old(
                self,
            ).received()[h.tag] == a.kind->Some_0,
        ensures
            Self::send_preserved(*old(self), *final(self), r, (a.share->Some_0, a.extra->Some_0)),
            final(self).sent() == (if r.is_ok() && h.grant() {
                old(self).sent().insert(h.tag, a.kind->Some_0)
            } else {
                old(self).sent()
            }),
            final(self).received() == (if r.is_ok() && !h.grant() && h.op < NOTICE {
                old(self).received().remove(h.tag)
            } else {
                old(self).received()
            }),
    {
        match self.send(h, user, Tracked(a)) {
            Ok(()) => Ok(()),
            Err((err, _ret)) => {
                let tracked recovered = match _ret.get() {
                    Some(att) => Some((att.share.tracked_unwrap(), att.extra.tracked_unwrap())),
                    None => None,
                };
                Err((err, Tracked(recovered)))
            },
        }
    }

    fn send_with_extra(&mut self, h: Header, user: &[u8], Tracked(a): Tracked<Attachment<M>>) -> (r:
        Result<(), (ChannelError<T::Error>, Tracked<Option<M>>)>)
        requires
            a.valid(old(self).id(), old(self).local()),
            a.header == h,
            a.user == user@,
            a.extra is Some,
            h.grant() ==> !old(self).sent().contains_key(h.tag),
            !h.grant() && h.op < NOTICE ==> old(self).received().contains_key(h.tag) && old(
                self,
            ).received()[h.tag] == a.kind->Some_0,
        ensures
            Self::send_preserved(*old(self), *final(self), r, a.extra->Some_0),
            final(self).sent() == (if r.is_ok() && h.grant() {
                old(self).sent().insert(h.tag, a.kind->Some_0)
            } else {
                old(self).sent()
            }),
            final(self).received() == (if r.is_ok() && !h.grant() && h.op < NOTICE {
                old(self).received().remove(h.tag)
            } else {
                old(self).received()
            }),
    {
        match self.send(h, user, Tracked(a)) {
            Ok(()) => Ok(()),
            Err((err, _ret)) => {
                let tracked recovered = match _ret.get() {
                    Some(att) => att.extra,
                    None => None,
                };
                Err((err, Tracked(recovered)))
            },
        }
    }

    fn recv_endpoint(&mut self, Tracked(instance): Tracked<&TransferInstance>) -> (r: Result<
        QueuePairEndpoint,
        ChannelError<T::Error>,
    >)
        requires
            old(self).id() == instance.id(),
        ensures
            final(self).same_session(*old(self)),
            r.is_ok() ==> !final(self).closed(),
            r.is_err() ==> final(self).closed(),
            final(self).sent() == old(self).sent(),
            final(self).received() == old(self).received(),
            r matches Ok(e) ==> e.node() == old(self).peer(),
    {
        if self.closed {
            return Err(ChannelError::Closed);
        }
        let (h, user_bytes, Tracked(a)) = match ffi::recv::<T, M>(
            &mut self.transport,
            M::magic_exec(),
            Ghost(self.instance_id@),
            self.local_node,
            self.peer_node,
            self.sent_grants,
            self.received_grants,
        ) {
            Ok(Some(v)) => v,
            Ok(None) => {
                self.closed = true;
                return Err(ChannelError::InvalidFrame);
            },
            Err(e) => {
                self.closed = true;
                return Err(ChannelError::Transport(e));
            },
        };
        if h.op != ENDPOINT {
            self.closed = true;
            return Err(ChannelError::InvalidFrame);
        }
        match decode_endpoint(&user_bytes, Ghost(a.endpoint->Some_0)) {
            Some(peer) => Ok(peer),
            None => {
                self.closed = true;
                Err(ChannelError::InvalidFrame)
            },
        }
    }

    fn recv_matching(
        &mut self,
        expected: Option<u8>,
        Tracked(instance): Tracked<&TransferInstance>,
    ) -> (r: Result<(u8, Received<M>), ChannelError<T::Error>>)
        requires
            old(self).id() == instance.id(),
            expected matches Some(wanted) ==> READ_GRANT <= wanted <= NOTICE,
        ensures
            final(self).same_session(*old(self)),
            r.is_ok() ==> !final(self).closed(),
            r.is_err() ==> final(self).closed() && final(self).sent() == old(self).sent()
                && final(self).received() == old(self).received(),
            r matches Ok((op, v)) ==> {
                &&& match expected {
                    Some(wanted) => op == wanted,
                    None => READ_GRANT <= op <= NOTICE,
                }
                &&& Self::delivery(*old(self), *final(self), op, v)
            },
    {
        if self.closed {
            return Err(ChannelError::Closed);
        }
        let (h, user_bytes, Tracked(a)) = match ffi::recv::<T, M>(
            &mut self.transport,
            M::magic_exec(),
            Ghost(self.instance_id@),
            self.local_node,
            self.peer_node,
            self.sent_grants,
            self.received_grants,
        ) {
            Ok(Some(v)) => v,
            Ok(None) => {
                self.closed = true;
                return Err(ChannelError::InvalidFrame);
            },
            Err(e) => {
                self.closed = true;
                return Err(ChannelError::Transport(e));
            },
        };
        let op = h.op;
        let accepted = match expected {
            Some(wanted) => op == wanted,
            None => READ_GRANT <= op && op <= NOTICE,
        };
        if !accepted {
            self.closed = true;
            return Err(ChannelError::InvalidFrame);
        }
        let capability = if op == READ_GRANT || op == WRITE_GRANT {
            Some(RemoteCapability::from_wire(h.addr, h.len, h.rkey, Ghost(a.cap->Some_0)))
        } else {
            None
        };
        let tracked s = a.statement;
        let ghost orig_s = s;
        let tracked knows = instance.learn(
            self.local(),
            self.peer(),
            s.facts,
            &s.submitted,
            &s.knows,
        );
        let tracked new_statement = Statement { knows, ..s };
        let tracked extra = a.extra.tracked_unwrap();
        proof {
            if h.grant() {
                self.received_grants = Ghost(self.received().insert(h.tag, a.kind->Some_0));
            } else if op < NOTICE {
                self.sent_grants = Ghost(self.sent().remove(h.tag));
            }
        }
        let received = Received {
            endpoint: Ghost(a.endpoint),
            tag: h.tag,
            user_bytes,
            capability,
            kind: Ghost(a.kind),
            slot: Tracked(a.slot),
            share: Tracked(a.share),
            statement: Tracked(new_statement),
            attachment: Tracked(extra),
        };
        proof {
            assert(orig_s.submitted == received.statement@.submitted);
            assert(orig_s.succeeded == received.statement@.succeeded);
            assert(orig_s.facts == received.statement@.facts);
            assert(received.attachment@.valid(
                old(self).id(),
                old(self).peer(),
                op,
                received.tag,
                received.user_bytes@,
                received.kind@,
                received.capability,
                share_frontier(received.share@),
                orig_s,
            ));
        }
        Ok((op, received))
    }

    fn recv(&mut self, op: u8, Tracked(instance): Tracked<&TransferInstance>) -> (r: Result<
        Received<M>,
        ChannelError<T::Error>,
    >)
        requires
            old(self).id() == instance.id(),
            READ_GRANT <= op <= NOTICE,
        ensures
            final(self).same_session(*old(self)),
            r.is_ok() ==> !final(self).closed(),
            r.is_err() ==> final(self).closed() && final(self).sent() == old(self).sent()
                && final(self).received() == old(self).received(),
            r matches Ok(v) ==> Self::delivery(*old(self), *final(self), op, v),
    {
        let (_, received) = self.recv_matching(Some(op), Tracked(instance))?;
        Ok(received)
    }

    /// Receive the next grant, release, or completion notice on this session.
    pub fn recv_any(&mut self, Tracked(instance): Tracked<&TransferInstance>) -> (r: Result<
        ReceivedFrame<M>,
        ChannelError<T::Error>,
    >)
        requires
            old(self).id() == instance.id(),
        ensures
            final(self).same_session(*old(self)),
            r.is_ok() ==> !final(self).closed(),
            r.is_err() ==> final(self).closed() && final(self).sent() == old(self).sent()
                && final(self).received() == old(self).received(),
            r matches Ok(frame) ==> match frame {
                ReceivedFrame::ReadGrant(v) => Self::delivery(*old(self), *final(self), 1, v),
                ReceivedFrame::ReadRelease(v) => Self::delivery(*old(self), *final(self), 2, v),
                ReceivedFrame::WriteGrant(v) => Self::delivery(*old(self), *final(self), 3, v),
                ReceivedFrame::WriteRelease(v) => Self::delivery(*old(self), *final(self), 4, v),
                ReceivedFrame::Notice(v) => Self::delivery(*old(self), *final(self), 5, v),
            },
    {
        let (op, received) = self.recv_matching(None, Tracked(instance))?;
        Ok(
            match op {
                READ_GRANT => ReceivedFrame::ReadGrant(received),
                READ_RELEASE => ReceivedFrame::ReadRelease(received),
                WRITE_GRANT => ReceivedFrame::WriteGrant(received),
                WRITE_RELEASE => ReceivedFrame::WriteRelease(received),
                _ => ReceivedFrame::Notice(received),
            },
        )
    }

    pub open spec fn delivery(before: Self, after: Self, op: u8, v: Received<M>) -> bool {
        let k = v.kind@;
        let grant = op == 1 || op == 3;
        &&& v.endpoint@ is None
        &&& v.learned(before.id(), before.local())
        &&& exists|s: Statement|
            s.submitted == v.statement@.submitted && s.succeeded == v.statement@.succeeded
                && s.facts == v.statement@.facts && #[trigger] v.attachment@.valid(
                before.id(),
                before.peer(),
                op,
                v.tag,
                v.user_bytes@,
                v.kind@,
                v.capability,
                share_frontier(v.share@),
                s,
            )
        &&& if grant {
            !before.received().contains_key(v.tag) && after.received() == before.received().insert(
                v.tag,
                k->Some_0,
            ) && after.sent() == before.sent()
        } else if op < 5 {
            before.sent().contains_key(v.tag) && k == Some(before.sent()[v.tag]) && after.sent()
                == before.sent().remove(v.tag) && after.received() == before.received()
        } else {
            after.sent() == before.sent() && after.received() == before.received()
        }
        &&& if op == 1 || op == 2 {
            k matches Some(GrantKind::ReadShare { slot, share_id }) && v.slot@ is None && (
            v.share@ matches Some(sh) && share_matches(
                sh,
                before.id(),
                slot,
                share_id,
                v.statement@.closure(),
            ))
        } else if op == 3 || op == 4 {
            k matches Some(GrantKind::OwnedSlot { slot }) && v.share@ is None && (
            v.slot@ matches Some(tok) && slot_matches(
                tok,
                before.id(),
                slot,
                v.statement@.closure(),
            ))
        } else {
            k is None && v.slot@ is None && v.share@ is None
        }
        &&& if grant {
            v.capability matches Some(c) && c.well_formed() && c.slot_ref().loc.node
                == before.peer() && (if op == 1 {
                c.share() == Some(k->Some_0->ReadShare_share_id) && c.slot_ref()
                    == k->Some_0->ReadShare_slot && c.access() == RemoteAccess::ReadOnly
            } else {
                c.slot_ref() == k->Some_0->OwnedSlot_slot && c.access() == RemoteAccess::ReadWrite
            })
        } else {
            v.capability is None
        }
    }

    pub fn recv_remote_read_grant(&mut self, Tracked(instance): Tracked<&TransferInstance>) -> (r:
        Result<Received<M>, ChannelError<T::Error>>)
        requires
            old(self).id() == instance.id(),
        ensures
            final(self).same_session(*old(self)),
            r.is_ok() ==> !final(self).closed(),
            r matches Ok(v) ==> Self::delivery(*old(self), *final(self), 1, v),
            r.is_err() ==> final(self).closed() && final(self).received() == old(self).received()
                && final(self).sent() == old(self).sent(),
    {
        self.recv(READ_GRANT, Tracked(instance))
    }

    pub fn recv_remote_read_release(&mut self, Tracked(instance): Tracked<&TransferInstance>) -> (r:
        Result<Received<M>, ChannelError<T::Error>>)
        requires
            old(self).id() == instance.id(),
        ensures
            final(self).same_session(*old(self)),
            r.is_ok() ==> !final(self).closed(),
            r matches Ok(v) ==> Self::delivery(*old(self), *final(self), 2, v),
            r.is_err() ==> final(self).closed() && final(self).received() == old(self).received()
                && final(self).sent() == old(self).sent(),
    {
        self.recv(READ_RELEASE, Tracked(instance))
    }

    pub fn recv_remote_write_grant(&mut self, Tracked(instance): Tracked<&TransferInstance>) -> (r:
        Result<Received<M>, ChannelError<T::Error>>)
        requires
            old(self).id() == instance.id(),
        ensures
            final(self).same_session(*old(self)),
            r.is_ok() ==> !final(self).closed(),
            r matches Ok(v) ==> Self::delivery(*old(self), *final(self), 3, v),
            r.is_err() ==> final(self).closed() && final(self).received() == old(self).received()
                && final(self).sent() == old(self).sent(),
    {
        self.recv(WRITE_GRANT, Tracked(instance))
    }

    pub fn recv_remote_write_release(
        &mut self,
        Tracked(instance): Tracked<&TransferInstance>,
    ) -> (r: Result<Received<M>, ChannelError<T::Error>>)
        requires
            old(self).id() == instance.id(),
        ensures
            final(self).same_session(*old(self)),
            r.is_ok() ==> !final(self).closed(),
            r matches Ok(v) ==> Self::delivery(*old(self), *final(self), 4, v),
            r.is_err() ==> final(self).closed() && final(self).received() == old(self).received()
                && final(self).sent() == old(self).sent(),
    {
        self.recv(WRITE_RELEASE, Tracked(instance))
    }

    pub fn recv_completion_notice(&mut self, Tracked(instance): Tracked<&TransferInstance>) -> (r:
        Result<Received<M>, ChannelError<T::Error>>)
        requires
            old(self).id() == instance.id(),
        ensures
            final(self).same_session(*old(self)),
            r.is_ok() ==> !final(self).closed(),
            r matches Ok(v) ==> Self::delivery(*old(self), *final(self), 5, v),
            r.is_err() ==> final(self).closed() && final(self).received() == old(self).received()
                && final(self).sent() == old(self).sent(),
    {
        self.recv(NOTICE, Tracked(instance))
    }

    /// Both peers advertise their own QP before receiving the peer's address.
    pub fn exchange_queue_pair_endpoint(
        &mut self,
        endpoint: &QueuePairEndpoint,
        Tracked(instance): Tracked<&TransferInstance>,
    ) -> (r: Result<QueuePairEndpoint, ChannelError<T::Error>>)
        requires
            old(self).id() == instance.id(),
            endpoint.node() == old(self).local(),
        ensures
            final(self).same_session(*old(self)),
            r.is_ok() ==> !final(self).closed(),
            r.is_err() ==> final(self).closed(),
            final(self).sent() == old(self).sent(),
            final(self).received() == old(self).received(),
            r matches Ok(e) ==> e.node() == old(self).peer(),
    {
        let h = Header { op: ENDPOINT, tag: 0, addr: 0, len: 0, rkey: 0 };
        let bytes = encode_endpoint(endpoint);
        let tracked s = Statement::empty(instance.id());
        let tracked a = Attachment {
            endpoint: Some(*endpoint),
            header: h,
            user: bytes@,
            kind: None,
            cap: None,
            statement: s,
            slot: None,
            share: None,
            extra: None,
        };
        if let Err((e, _)) = self.send(h, &bytes, Tracked(a)) {
            return Err(e);
        }
        self.recv_endpoint(Tracked(instance))
    }

    pub fn send_remote_read_grant_with(
        &mut self,
        tag: GrantTag,
        cap: &RemoteCapability,
        user: &[u8],
        Tracked(instance): Tracked<&TransferInstance>,
        Tracked(submitted): Tracked<&SubmittedMap>,
        Tracked(knows): Tracked<&Knows>,
        Tracked(succeeded): Tracked<&SucceededMap>,
        Ghost(facts): Ghost<ISet<TransferEvent>>,
        Tracked(share): Tracked<ShareCap>,
        Tracked(extra): Tracked<M>,
    ) -> (r: Result<(), (ChannelError<T::Error>, Tracked<Option<(ShareCap, M)>>)>)
        requires
            instance.id() == old(self).id(),
            submitted.instance_id() == old(self).id(),
            knows.instance_id() == old(self).id(),
            succeeded.instance_id() == old(self).id(),
            succeeded.well_formed(),
            facts.subset_of(submitted.dom()),
            knowledge_of(old(self).local(), facts).subset_of(knows.set()),
            forall|e| #[trigger] succeeded.contains(e) ==> submitted.dom().contains(e),
            cap.well_formed(),
            cap.slot_ref().loc.node == old(self).local(),
            cap.access() == RemoteAccess::ReadOnly,
            cap.share() == Some(share.share()),
            !old(self).sent().contains_key(tag),
            share_matches(
                share,
                old(self).id(),
                cap.slot_ref(),
                share.share(),
                closure_from(submitted.map(), facts),
            ),
            forall|s: Statement|
                s.submitted.instance_id() == old(self).id() && s.submitted.map() == submitted.map()
                    && s.knows.instance_id() == old(self).id() && s.knows.set() == knows.set()
                    && s.succeeded == *succeeded && s.facts == facts ==> #[trigger] extra.valid(
                    old(self).id(),
                    old(self).local(),
                    1,
                    tag,
                    user@,
                    Some(GrantKind::ReadShare { slot: cap.slot_ref(), share_id: share.share() }),
                    Some(*cap),
                    Some(share.frontier()),
                    s,
                ),
        ensures
            Self::send_preserved(*old(self), *final(self), r, (share, extra)),
            final(self).sent() == (if r.is_ok() {
                old(self).sent().insert(
                    tag,
                    GrantKind::ReadShare { slot: cap.slot_ref(), share_id: share.share() },
                )
            } else {
                old(self).sent()
            }),
            final(self).received() == old(self).received(),
    {
        let (addr, len, rkey) = cap.wire_fields();
        let h = Header { op: READ_GRANT, tag, addr, len, rkey };
        let ghost share_id = share.share();
        let tracked s = Statement {
            succeeded: succeeded.duplicate(),
            submitted: instance.duplicate_submitted(submitted),
            knows: instance.duplicate_knows(knows),
            facts,
        };
        let tracked a = Attachment {
            endpoint: None,
            header: h,
            user: user@,
            kind: Some(GrantKind::ReadShare { slot: cap.slot_ref(), share_id }),
            cap: Some(*cap),
            statement: s,
            slot: None,
            share: Some(share),
            extra: Some(extra),
        };
        self.send_with_share_and_extra(h, user, Tracked(a))
    }

    pub fn send_remote_read_release_with(
        &mut self,
        tag: GrantTag,
        user: &[u8],
        Tracked(instance): Tracked<&TransferInstance>,
        Tracked(submitted): Tracked<&SubmittedMap>,
        Tracked(knows): Tracked<&Knows>,
        Tracked(succeeded): Tracked<&SucceededMap>,
        Ghost(facts): Ghost<ISet<TransferEvent>>,
        Tracked(share): Tracked<ShareCap>,
        Tracked(extra): Tracked<M>,
    ) -> (r: Result<(), (ChannelError<T::Error>, Tracked<Option<(ShareCap, M)>>)>)
        requires
            instance.id() == old(self).id(),
            submitted.instance_id() == old(self).id(),
            knows.instance_id() == old(self).id(),
            succeeded.instance_id() == old(self).id(),
            succeeded.well_formed(),
            facts.subset_of(submitted.dom()),
            knowledge_of(old(self).local(), facts).subset_of(knows.set()),
            forall|e| #[trigger] succeeded.contains(e) ==> submitted.dom().contains(e),
            old(self).received().contains_key(tag),
            old(self).received()[tag] is ReadShare,
            share_matches(
                share,
                old(self).id(),
                old(self).received()[tag]->ReadShare_slot,
                old(self).received()[tag]->ReadShare_share_id,
                closure_from(submitted.map(), facts),
            ),
            forall|s: Statement|
                s.submitted.instance_id() == old(self).id() && s.submitted.map() == submitted.map()
                    && s.knows.instance_id() == old(self).id() && s.knows.set() == knows.set()
                    && s.succeeded == *succeeded && s.facts == facts ==> #[trigger] extra.valid(
                    old(self).id(),
                    old(self).local(),
                    2,
                    tag,
                    user@,
                    Some(old(self).received()[tag]),
                    None,
                    Some(share.frontier()),
                    s,
                ),
        ensures
            Self::send_preserved(*old(self), *final(self), r, (share, extra)),
            final(self).sent() == old(self).sent(),
            final(self).received() == (if r.is_ok() {
                old(self).received().remove(tag)
            } else {
                old(self).received()
            }),
    {
        let h = Header { op: READ_RELEASE, tag, addr: 0, len: 0, rkey: 0 };
        let tracked s = Statement {
            succeeded: succeeded.duplicate(),
            submitted: instance.duplicate_submitted(submitted),
            knows: instance.duplicate_knows(knows),
            facts,
        };
        let tracked a = Attachment {
            endpoint: None,
            header: h,
            user: user@,
            kind: Some(self.received()[tag]),
            cap: None,
            statement: s,
            slot: None,
            share: Some(share),
            extra: Some(extra),
        };
        self.send_with_share_and_extra(h, user, Tracked(a))
    }

    pub fn send_notice_with(
        &mut self,
        user: &[u8],
        Tracked(instance): Tracked<&TransferInstance>,
        Tracked(submitted): Tracked<&SubmittedMap>,
        Tracked(knows): Tracked<&Knows>,
        Tracked(succeeded): Tracked<&SucceededMap>,
        Ghost(facts): Ghost<ISet<TransferEvent>>,
        Tracked(extra): Tracked<M>,
    ) -> (r: Result<(), (ChannelError<T::Error>, Tracked<Option<M>>)>)
        requires
            instance.id() == old(self).id(),
            submitted.instance_id() == old(self).id(),
            knows.instance_id() == old(self).id(),
            succeeded.instance_id() == old(self).id(),
            succeeded.well_formed(),
            facts.subset_of(submitted.dom()),
            knowledge_of(old(self).local(), facts).subset_of(knows.set()),
            forall|e| #[trigger] succeeded.contains(e) ==> submitted.dom().contains(e),
            forall|s: Statement|
                s.submitted.instance_id() == old(self).id() && s.submitted.map() == submitted.map()
                    && s.knows.instance_id() == old(self).id() && s.knows.set() == knows.set()
                    && s.succeeded == *succeeded && s.facts == facts ==> #[trigger] extra.valid(
                    old(self).id(),
                    old(self).local(),
                    5,
                    0,
                    user@,
                    None,
                    None,
                    None,
                    s,
                ),
        ensures
            Self::send_preserved(*old(self), *final(self), r, extra),
            final(self).sent() == old(self).sent(),
            final(self).received() == old(self).received(),
    {
        let h = Header { op: NOTICE, tag: 0, addr: 0, len: 0, rkey: 0 };
        let tracked s = Statement {
            succeeded: succeeded.duplicate(),
            submitted: instance.duplicate_submitted(submitted),
            knows: instance.duplicate_knows(knows),
            facts,
        };
        let tracked a = Attachment {
            endpoint: None,
            header: h,
            user: user@,
            kind: None,
            cap: None,
            statement: s,
            slot: None,
            share: None,
            extra: Some(extra),
        };
        self.send_with_extra(h, user, Tracked(a))
    }
}

impl<T: ByteTransport> Channel<T, ()> {
    pub fn new(
        transport: T,
        local_node: NodeId,
        peer_node: NodeId,
        Tracked(instance): Tracked<&TransferInstance>,
    ) -> (c: Self)
        requires
            local_node != peer_node,
        ensures
            c.local() == local_node && c.peer() == peer_node && c.id() == instance.id() && c.sent()
                == Map::empty() && c.received() == Map::empty() && !c.closed(),
    {
        Self {
            transport,
            local_node,
            peer_node,
            instance_id: Ghost(instance.id()),
            sent_grants: Ghost(Map::empty()),
            received_grants: Ghost(Map::empty()),
            closed: false,
            _marker: core::marker::PhantomData,
        }
    }

    fn send_with_share(
        &mut self,
        h: Header,
        user: &[u8],
        Tracked(a): Tracked<Attachment<()>>,
    ) -> (r: Result<(), (ChannelError<T::Error>, Tracked<Option<ShareCap>>)>)
        requires
            a.valid(old(self).id(), old(self).local()),
            a.header == h,
            a.user == user@,
            a.share is Some,
            h.grant() ==> !old(self).sent().contains_key(h.tag),
            !h.grant() && h.op < NOTICE ==> old(self).received().contains_key(h.tag) && old(
                self,
            ).received()[h.tag] == a.kind->Some_0,
        ensures
            Self::send_preserved(*old(self), *final(self), r, a.share->Some_0),
            final(self).sent() == (if r.is_ok() && h.grant() {
                old(self).sent().insert(h.tag, a.kind->Some_0)
            } else {
                old(self).sent()
            }),
            final(self).received() == (if r.is_ok() && !h.grant() && h.op < NOTICE {
                old(self).received().remove(h.tag)
            } else {
                old(self).received()
            }),
    {
        match self.send(h, user, Tracked(a)) {
            Ok(()) => Ok(()),
            Err((err, _ret)) => {
                let tracked recovered = match _ret.get() {
                    Some(att) => att.share,
                    None => None,
                };
                Err((err, Tracked(recovered)))
            },
        }
    }

    fn send_with_slot(&mut self, h: Header, user: &[u8], Tracked(a): Tracked<Attachment<()>>) -> (r:
        Result<(), (ChannelError<T::Error>, Tracked<Option<SlotToken>>)>)
        requires
            a.valid(old(self).id(), old(self).local()),
            a.header == h,
            a.user == user@,
            a.slot is Some,
            h.grant() ==> !old(self).sent().contains_key(h.tag),
            !h.grant() && h.op < NOTICE ==> old(self).received().contains_key(h.tag) && old(
                self,
            ).received()[h.tag] == a.kind->Some_0,
        ensures
            Self::send_preserved(*old(self), *final(self), r, a.slot->Some_0),
            final(self).sent() == (if r.is_ok() && h.grant() {
                old(self).sent().insert(h.tag, a.kind->Some_0)
            } else {
                old(self).sent()
            }),
            final(self).received() == (if r.is_ok() && !h.grant() && h.op < NOTICE {
                old(self).received().remove(h.tag)
            } else {
                old(self).received()
            }),
    {
        match self.send(h, user, Tracked(a)) {
            Ok(()) => Ok(()),
            Err((err, _ret)) => {
                let tracked recovered = match _ret.get() {
                    Some(att) => att.slot,
                    None => None,
                };
                Err((err, Tracked(recovered)))
            },
        }
    }

    pub fn send_remote_read_grant(
        &mut self,
        tag: GrantTag,
        cap: &RemoteCapability,
        user: &[u8],
        Tracked(s): Tracked<Statement>,
        Tracked(share): Tracked<ShareCap>,
    ) -> (r: Result<(), (ChannelError<T::Error>, Tracked<Option<ShareCap>>)>)
        requires
            s.valid(old(self).id(), old(self).local()),
            cap.well_formed(),
            cap.slot_ref().loc.node == old(self).local(),
            cap.access() == RemoteAccess::ReadOnly,
            cap.share() == Some(share.share()),
            !old(self).sent().contains_key(tag),
            share_matches(share, old(self).id(), cap.slot_ref(), share.share(), s.closure()),
        ensures
            Self::send_preserved(*old(self), *final(self), r, share),
            final(self).sent() == (if r.is_ok() {
                old(self).sent().insert(
                    tag,
                    GrantKind::ReadShare { slot: cap.slot_ref(), share_id: share.share() },
                )
            } else {
                old(self).sent()
            }),
            final(self).received() == old(self).received(),
    {
        let (addr, len, rkey) = cap.wire_fields();
        let h = Header { op: READ_GRANT, tag, addr, len, rkey };
        let ghost share_id = share.share();
        let tracked a = Attachment {
            endpoint: None,
            header: h,
            user: user@,
            kind: Some(GrantKind::ReadShare { slot: cap.slot_ref(), share_id }),
            cap: Some(*cap),
            statement: s,
            slot: None,
            share: Some(share),
            extra: Some(()),
        };
        self.send_with_share(h, user, Tracked(a))
    }

    pub fn send_remote_read_release(
        &mut self,
        tag: GrantTag,
        user: &[u8],
        Tracked(s): Tracked<Statement>,
        Tracked(share): Tracked<ShareCap>,
    ) -> (r: Result<(), (ChannelError<T::Error>, Tracked<Option<ShareCap>>)>)
        requires
            s.valid(old(self).id(), old(self).local()),
            old(self).received().contains_key(tag),
            old(self).received()[tag] is ReadShare,
            share_matches(
                share,
                old(self).id(),
                old(self).received()[tag]->ReadShare_slot,
                old(self).received()[tag]->ReadShare_share_id,
                s.closure(),
            ),
        ensures
            Self::send_preserved(*old(self), *final(self), r, share),
            final(self).sent() == old(self).sent(),
            final(self).received() == (if r.is_ok() {
                old(self).received().remove(tag)
            } else {
                old(self).received()
            }),
    {
        let h = Header { op: READ_RELEASE, tag, addr: 0, len: 0, rkey: 0 };
        let tracked a = Attachment {
            endpoint: None,
            header: h,
            user: user@,
            kind: Some(self.received()[tag]),
            cap: None,
            statement: s,
            slot: None,
            share: Some(share),
            extra: Some(()),
        };
        self.send_with_share(h, user, Tracked(a))
    }

    pub fn send_remote_write_grant(
        &mut self,
        tag: GrantTag,
        cap: &RemoteCapability,
        user: &[u8],
        Tracked(s): Tracked<Statement>,
        Tracked(slot): Tracked<SlotToken>,
    ) -> (r: Result<(), (ChannelError<T::Error>, Tracked<Option<SlotToken>>)>)
        requires
            s.valid(old(self).id(), old(self).local()),
            cap.well_formed(),
            cap.slot_ref().loc.node == old(self).local(),
            cap.access() == RemoteAccess::ReadWrite,
            !old(self).sent().contains_key(tag),
            slot_matches(slot, old(self).id(), cap.slot_ref(), s.closure()),
        ensures
            Self::send_preserved(*old(self), *final(self), r, slot),
            final(self).sent() == (if r.is_ok() {
                old(self).sent().insert(tag, GrantKind::OwnedSlot { slot: cap.slot_ref() })
            } else {
                old(self).sent()
            }),
            final(self).received() == old(self).received(),
    {
        let (addr, len, rkey) = cap.wire_fields();
        let h = Header { op: WRITE_GRANT, tag, addr, len, rkey };
        let tracked a = Attachment {
            endpoint: None,
            header: h,
            user: user@,
            kind: Some(GrantKind::OwnedSlot { slot: cap.slot_ref() }),
            cap: Some(*cap),
            statement: s,
            slot: Some(slot),
            share: None,
            extra: Some(()),
        };
        self.send_with_slot(h, user, Tracked(a))
    }

    pub fn send_remote_write_release(
        &mut self,
        tag: GrantTag,
        user: &[u8],
        Tracked(s): Tracked<Statement>,
        Tracked(slot): Tracked<SlotToken>,
    ) -> (r: Result<(), (ChannelError<T::Error>, Tracked<Option<SlotToken>>)>)
        requires
            s.valid(old(self).id(), old(self).local()),
            old(self).received().contains_key(tag),
            old(self).received()[tag] is OwnedSlot,
            slot_matches(
                slot,
                old(self).id(),
                old(self).received()[tag]->OwnedSlot_slot,
                s.closure(),
            ),
        ensures
            Self::send_preserved(*old(self), *final(self), r, slot),
            final(self).sent() == old(self).sent(),
            final(self).received() == (if r.is_ok() {
                old(self).received().remove(tag)
            } else {
                old(self).received()
            }),
    {
        let h = Header { op: WRITE_RELEASE, tag, addr: 0, len: 0, rkey: 0 };
        let tracked a = Attachment {
            endpoint: None,
            header: h,
            user: user@,
            kind: Some(self.received()[tag]),
            cap: None,
            statement: s,
            slot: Some(slot),
            share: None,
            extra: Some(()),
        };
        self.send_with_slot(h, user, Tracked(a))
    }

    pub fn send_completion_notice(&mut self, user: &[u8], Tracked(s): Tracked<Statement>) -> (r:
        Result<(), ChannelError<T::Error>>)
        requires
            s.valid(old(self).id(), old(self).local()),
        ensures
            final(self).same_session(*old(self)),
            r.is_ok() ==> !final(self).closed(),
            r.is_err() ==> final(self).closed(),
            final(self).sent() == old(self).sent(),
            final(self).received() == old(self).received(),
    {
        let h = Header { op: NOTICE, tag: 0, addr: 0, len: 0, rkey: 0 };
        let tracked a = Attachment {
            endpoint: None,
            header: h,
            user: user@,
            kind: None,
            cap: None,
            statement: s,
            slot: None,
            share: None,
            extra: Some(()),
        };
        match self.send(h, user, Tracked(a)) {
            Ok(()) => Ok(()),
            Err((e, _)) => Err(e),
        }
    }
}

/// A reader can return a grant with a success receipt. A peer can consume that
/// witness from either a release or a notice through the same receive API.
#[cfg(verus_only)]
fn example_read_dispatch<T: ByteTransport>(
    channel: &mut Channel<T>,
    Tracked(instance): Tracked<&TransferInstance>,
    Tracked(success): Tracked<Succeeded>,
)
    requires
        old(channel).id() == instance.id(),
        success.instance_id() == instance.id(),
{
    let ghost sent = channel.sent();
    let Ok(frame) = channel.recv_any(Tracked(instance)) else { return };
    let received = match frame {
        ReceivedFrame::ReadGrant(grant) => {
            let Tracked(mut statement) = grant.statement;
            proof {
                assert(statement.valid(channel.id(), channel.local()));
                if statement.submitted.dom().contains(success.event()) {
                    statement.succeeded.insert(success);
                    assert(statement.succeeded.contains(success.event()));
                    assert forall|event| #[trigger]
                        statement.succeeded.contains(
                            event,
                        ) implies statement.submitted.dom().contains(event) by {
                        if event != success.event() {
                            assert(grant.statement@.succeeded.contains(event));
                        }
                    }
                }
            }
            let _ = channel.send_remote_read_release(
                grant.tag,
                &grant.user_bytes,
                Tracked(statement),
                Tracked(grant.share.get().tracked_unwrap()),
            );
            return;
        },
        ReceivedFrame::ReadRelease(release) => {
            assert(release.kind@ == Some(sent[release.tag]));
            assert(channel.sent() == sent.remove(release.tag));
            release
        },
        ReceivedFrame::Notice(notice) => {
            assert(channel.sent() == sent);
            notice
        },
        _ => return,
    };
    proof {
        let tracked statement = received.statement.get();
        let event = success.event();
        if statement.succeeded.contains(event) {
            let tracked witness = statement.succeeded.duplicate_for(event);
            assert(statement.submitted.dom().contains(event));
            assert(witness.instance_id() == instance.id() && witness.event() == event);
        }
    }
}

} // verus!
#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;

    struct ClosedTestTransport {
        send_calls: usize,
        recv_frame: Option<Vec<u8>>,
    }

    impl ByteTransport for ClosedTestTransport {
        type Error = &'static str;

        fn send_frame(&mut self, _: &[u8]) -> Result<(), Self::Error> {
            self.send_calls += 1;
            Err("transport send failed")
        }

        fn recv_frame(&mut self) -> Result<Vec<u8>, Self::Error> {
            self.recv_frame.take().ok_or("transport recv failed")
        }
    }

    struct LoopbackTransport {
        frames: VecDeque<Vec<u8>>,
    }

    impl LoopbackTransport {
        fn new() -> Self {
            Self {
                frames: VecDeque::new(),
            }
        }
    }

    impl ByteTransport for LoopbackTransport {
        type Error = &'static str;

        fn send_frame(&mut self, frame: &[u8]) -> Result<(), Self::Error> {
            self.frames.push_back(frame.to_vec());
            Ok(())
        }

        fn recv_frame(&mut self) -> Result<Vec<u8>, Self::Error> {
            self.frames.pop_front().ok_or("no frame queued")
        }
    }

    const TEST_MAGIC_A: u32 = 0x5643_0001;
    const TEST_MAGIC_B: u32 = 0x5643_0002;

    struct TestProtoA;

    impl ChannelAttachment for TestProtoA {
        fn magic_exec() -> u32 {
            TEST_MAGIC_A
        }
    }

    #[test]
    fn closed_channel_rejects_send_before_transport() {
        let mut ch = Channel::new(
            ClosedTestTransport {
                send_calls: 0,
                recv_frame: Some(vec![7]),
            },
            0,
            1,
            Tracked::assume_new(),
        );
        assert!(!ch.is_closed());
        assert!(matches!(
            ch.recv_any(Tracked::assume_new()),
            Err(ChannelError::InvalidFrame)
        ));
        assert!(ch.is_closed());
        let res = ch.send(
            Header {
                op: NOTICE,
                tag: 0,
                addr: 0,
                len: 0,
                rkey: 0,
            },
            b"hi",
            Tracked::assume_new(),
        );
        assert!(matches!(res, Err((ChannelError::Closed, _))));
        assert_eq!(ch.transport.send_calls, 0);
    }

    #[test]
    fn protocol_magic_wire_and_channel_round_trip() {
        let h = Header {
            op: NOTICE,
            tag: 0,
            addr: 0,
            len: 0,
            rkey: 0,
        };
        let encoded = encode_with_magic(TEST_MAGIC_A, h, b"hello-proto");
        assert_eq!(encoded[0], PROTOCOL_MAGIC_TAG);
        assert_eq!(&encoded[1..5], &TEST_MAGIC_A.to_le_bytes());
        let (decoded_h, decoded_user) =
            decode_with_magic(TEST_MAGIC_A, &encoded).expect("magic frame should decode");
        assert_eq!(decoded_h.op, NOTICE);
        assert_eq!(decoded_user, b"hello-proto");
        assert!(decode_with_magic(0, &encoded).is_none());
        assert!(decode_with_magic(TEST_MAGIC_B, &encoded).is_none());

        let mut sender = Channel::<LoopbackTransport, TestProtoA>::new_with_protocol(
            LoopbackTransport::new(),
            0,
            1,
            Tracked::assume_new(),
        );
        sender
            .send_notice_with(
                b"proto-payload",
                Tracked::assume_new(),
                Tracked::assume_new(),
                Tracked::assume_new(),
                Tracked::assume_new(),
                Ghost::assume_new(),
                Tracked::assume_new(),
            )
            .expect("send_notice_with should succeed");

        let queued = sender.transport.frames.pop_front().expect("frame queued");
        let mut receiver = Channel::<LoopbackTransport, TestProtoA>::new_with_protocol(
            LoopbackTransport {
                frames: VecDeque::from([queued]),
            },
            1,
            0,
            Tracked::assume_new(),
        );
        match receiver
            .recv_any(Tracked::assume_new())
            .expect("recv_any should decode protocol frame")
        {
            ReceivedFrame::Notice(msg) => assert_eq!(msg.user_bytes, b"proto-payload"),
            _ => panic!("expected Notice frame"),
        }
        assert!(!receiver.is_closed());
    }

    #[test]
    fn mismatched_magic_and_legacy_frames_rejected() {
        let h = Header {
            op: NOTICE,
            tag: 0,
            addr: 0,
            len: 0,
            rkey: 0,
        };

        // 1. Wrong non-zero protocol magic rejected by TestProtoA channel.
        let wrong_magic_frame = encode_with_magic(TEST_MAGIC_B, h, b"wrong");
        let mut ch_wrong = Channel::<LoopbackTransport, TestProtoA>::new_with_protocol(
            LoopbackTransport {
                frames: VecDeque::from([wrong_magic_frame]),
            },
            1,
            0,
            Tracked::assume_new(),
        );
        assert!(matches!(
            ch_wrong.recv_any(Tracked::assume_new()),
            Err(ChannelError::InvalidFrame)
        ));
        assert!(ch_wrong.is_closed());

        // 2. Legacy unprefixed frame (magic == 0) rejected by TestProtoA channel.
        let legacy_frame = encode_with_magic(0, h, b"legacy");
        let mut ch_legacy_into_proto = Channel::<LoopbackTransport, TestProtoA>::new_with_protocol(
            LoopbackTransport {
                frames: VecDeque::from([legacy_frame]),
            },
            1,
            0,
            Tracked::assume_new(),
        );
        assert!(matches!(
            ch_legacy_into_proto.recv_any(Tracked::assume_new()),
            Err(ChannelError::InvalidFrame)
        ));
        assert!(ch_legacy_into_proto.is_closed());

        // 3. Magic-prefixed frame rejected by legacy () channel.
        let proto_frame = encode_with_magic(TEST_MAGIC_A, h, b"proto");
        let mut ch_proto_into_legacy = Channel::new(
            LoopbackTransport {
                frames: VecDeque::from([proto_frame]),
            },
            1,
            0,
            Tracked::assume_new(),
        );
        assert!(matches!(
            ch_proto_into_legacy.recv_any(Tracked::assume_new()),
            Err(ChannelError::InvalidFrame)
        ));
        assert!(ch_proto_into_legacy.is_closed());
    }
}
