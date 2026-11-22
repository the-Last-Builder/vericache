use super::*;

verus! {

broadcast use {
    vstd::imap::group_imap_lemmas,
    vstd::iset::group_iset_lemmas,
    vstd::map::group_map_lemmas,
};

pub enum PendingKind {
    DataTransfer { event: TransferEvent },
    BindWindow {
        qp_id: nat,
        share: Option<TransferShare>,
        window_id: nat,
        target_slot: SlotRef,
        addr: u64,
        next_rkey: u32,
        access: RemoteAccess,
    },
    LocalInvalidate { qp_id: nat, window_id: nat },
}

impl PendingKind {
    pub open spec fn qp_id(self) -> nat {
        match self {
            Self::DataTransfer { event } => event.source->RdmaQueuePair_id,
            Self::BindWindow { qp_id, .. } | Self::LocalInvalidate { qp_id, .. } => qp_id,
        }
    }
}

pub struct PendingWorkRequest {
    pub kind: PendingKind,
}

pub struct CompletionQueue<'a> {
    pub(super) raw: *mut u8,
    pub(super) device: &'a DeviceContext,
    pub(super) capacity: usize,
    pub(super) in_flight_count: usize,
    #[cfg_attr(not(verus_only), allow(dead_code))]
    pub(super) cq_id: Ghost<nat>,
    #[cfg_attr(not(verus_only), allow(dead_code))]
    pub(super) in_flight: Ghost<Map<u64, PendingWorkRequest>>,
    #[cfg_attr(not(verus_only), allow(dead_code))]
    pub(super) in_flight_infos: Tracked<SubmittedMap>,
}

impl<'a> CompletionQueue<'a> {
    pub open(super) spec fn id(&self) -> nat {
        self.cq_id@
    }

    pub open(super) spec fn observer(&self) -> NodeId {
        self.device.node()
    }

    pub open(super) spec fn pending(&self) -> Map<u64, PendingWorkRequest> {
        self.in_flight@
    }

    pub open spec fn in_flight(&self) -> Map<u64, PendingWorkRequest> {
        self.pending()
    }

    pub open(super) spec fn capacity(&self) -> nat {
        self.capacity as nat
    }

    pub fn capacity_exec(&self) -> (c: usize)
        ensures
            c as nat == self.capacity(),
    {
        self.capacity
    }

    pub fn in_flight_count(&self) -> (n: usize)
        requires
            self.well_formed(),
        ensures
            n as nat == self.in_flight().len(),
    {
        self.in_flight_count
    }

    pub open(super) spec fn infos(&self) -> SubmittedMap {
        self.in_flight_infos@
    }

    pub open(super) spec fn well_formed(&self) -> bool {
        &&& self.capacity > 0
        &&& self.in_flight_count <= self.capacity
        &&& self.in_flight_count as nat == self.pending().len()
        &&& forall|k: u64|
            self.pending().contains_key(k) && (#[trigger] self.pending()[k]).kind is DataTransfer
                ==> self.infos().dom().contains(self.pending()[k].kind->DataTransfer_event)
    }

    pub open spec fn same_queue(&self, before: Self) -> bool {
        self.id() == before.id() && self.observer() == before.observer() && self.capacity()
            == before.capacity() && self.infos().instance_id() == before.infos().instance_id()
    }

    pub fn new(
        device: &'a DeviceContext,
        depth: i32,
        Tracked(name): Tracked<EventSourceToken>,
    ) -> (r: Result<Self, RdmaError>)
        requires
            depth > 0,
            name.source() is RdmaQueuePair,
            name.well_formed(),
            name.node() == device.node(),
        ensures
            r matches Ok(c) ==> c.well_formed() && c.pending() == Map::empty() && c.capacity()
                == depth as nat && c.infos().instance_id() == name.id() && c.observer()
                == device.node() && c.id() == name.source()->RdmaQueuePair_id,
    {
        if depth <= 0 {
            return Err(RdmaError(22));
        }
        Ok(
            Self {
                raw: ffi::create_cq(device.raw, depth)?,
                device,
                capacity: depth as usize,
                in_flight_count: 0,
                cq_id: Ghost(name.source()->RdmaQueuePair_id),
                in_flight: Ghost(Map::empty()),
                in_flight_infos: Tracked(SubmittedMap::empty(name.id())),
            },
        )
    }

    pub fn destroy(self) -> Result<(), RdmaError>
        requires
            self.pending().is_empty(),
    {
        ffi::destroy_cq(self.raw)
    }
}

#[derive(Clone, Copy)]
pub struct QueuePairEndpoint {
    pub(super) number: u32,
    pub(super) psn: u32,
    pub(super) lid: u16,
    pub(super) gid: [u8; 16],
    pub(super) mtu: u8,
    pub(super) responder_resources: u8,
    #[cfg_attr(not(verus_only), allow(dead_code))]
    origin: Ghost<NodeId>,
}

impl QueuePairEndpoint {
    pub closed spec fn node(&self) -> NodeId {
        self.origin@
    }

    pub closed spec fn responder_resources_spec(&self) -> u8 {
        self.responder_resources
    }

    pub(crate) closed spec fn fields(&self) -> (u32, u32, u16, Seq<u8>, u8, u8) {
        (self.number, self.psn, self.lid, self.gid@, self.mtu, self.responder_resources)
    }

    pub(crate) fn components(&self) -> (r: (u32, u32, u16, [u8; 16], u8, u8))
        ensures
            self.fields() == (r.0, r.1, r.2, r.3@, r.4, r.5),
    {
        (self.number, self.psn, self.lid, self.gid, self.mtu, self.responder_resources)
    }

    pub(crate) fn restore(
        number: u32,
        psn: u32,
        lid: u16,
        gid: [u8; 16],
        mtu: u8,
        responder_resources: u8,
        Ghost(sent): Ghost<Self>,
    ) -> (r: Self)
        requires
            sent.fields() == (number, psn, lid, gid@, mtu, responder_resources),
        ensures
            r == sent,
    {
        assert(gid == sent.gid);
        Self { number, psn, lid, gid, mtu, responder_resources, origin: Ghost(sent.node()) }
    }
}

pub struct UnconnectedQueuePair<'a> {
    pub(super) raw: *mut u8,
    pub(super) pd: &'a ProtectionDomain<'a>,
    #[cfg_attr(not(verus_only), allow(dead_code))]
    pub(super) cq_id: Ghost<nat>,
    #[cfg_attr(not(verus_only), allow(dead_code))]
    pub(super) source: Tracked<EventSourceToken>,
    pub(super) endpoint: QueuePairEndpoint,
    pub(super) port: u8,
    pub(super) gid_index: u8,
    pub(super) global: bool,
    pub(super) config: QueuePairConfig,
}

pub enum QueuePairState {
    ReadyToSend,
    Error,
    Reset,
}

pub struct QueuePair<'a> {
    pub(super) connection: UnconnectedQueuePair<'a>,
    pub(super) effective_max_rd_atomic: u8,
    #[cfg_attr(not(verus_only), allow(dead_code))]
    pub(super) state: Ghost<QueuePairState>,
    #[cfg_attr(not(verus_only), allow(dead_code))]
    pub(super) peer_node: Ghost<NodeId>,
    #[cfg_attr(not(verus_only), allow(dead_code))]
    pub(super) pending_reads: Ghost<ISet<TransferEvent>>,
    #[cfg_attr(not(verus_only), allow(dead_code))]
    pub(super) pending_read_infos: Tracked<SubmittedMap>,
}

impl<'a> UnconnectedQueuePair<'a> {
    #[verifier::type_invariant]
    pub(super) open spec fn configured(&self) -> bool {
        self.config.send_depth > 0 && self.config.max_send_sge > 0
    }

    pub open(super) spec fn source(&self) -> EventSourceToken {
        self.source@
    }

    pub open(super) spec fn pd_id(&self) -> nat {
        self.pd.id()
    }

    pub open(super) spec fn cq_id(&self) -> nat {
        self.cq_id@
    }

    pub open(super) spec fn configuration(&self) -> QueuePairConfig {
        self.config
    }

    pub fn new(
        pd: &'a ProtectionDomain<'a>,
        cq: &CompletionQueue,
        depth: u32,
        port: u8,
        gid_index: u8,
        psn: u32,
        Tracked(source): Tracked<EventSourceToken>,
    ) -> (r: Result<Self, RdmaError>)
        requires
            source.well_formed(),
            source.source() is RdmaQueuePair,
            source.node() == pd.node(),
            source.node() == cq.observer(),
            source.id() == cq.infos().instance_id(),
        ensures
            r matches Ok(q) ==> q.source() == source && q.pd_id() == pd.id() && q.cq_id() == cq.id()
                && q.configuration().send_depth == depth && q.configuration().max_send_sge == 1,
    {
        Self::new_with_config(
            pd,
            cq,
            QueuePairConfig {
                send_depth: depth,
                max_send_sge: 1,
                max_rd_atomic: 1,
                max_dest_rd_atomic: 1,
            },
            port,
            gid_index,
            psn,
            Tracked(source),
        )
    }

    pub fn new_with_config(
        pd: &'a ProtectionDomain<'a>,
        cq: &CompletionQueue,
        config: QueuePairConfig,
        port: u8,
        gid_index: u8,
        psn: u32,
        Tracked(source): Tracked<EventSourceToken>,
    ) -> (r: Result<Self, RdmaError>)
        requires
            source.well_formed(),
            source.source() is RdmaQueuePair,
            source.node() == pd.node(),
            source.node() == cq.observer(),
            source.id() == cq.infos().instance_id(),
        ensures
            r matches Ok(q) ==> q.source() == source && q.pd_id() == pd.id() && q.cq_id() == cq.id()
                && q.configuration() == config,
    {
        if config.send_depth == 0 || config.max_send_sge == 0 {
            return Err(RdmaError(22));
        }
        let attr = pd.context.attributes()?;
        if config.send_depth > attr.max_qp_wr || config.max_send_sge > attr.max_send_sge || (
        config.max_rd_atomic > 0 && config.max_send_sge > attr.max_read_sge) || config.max_rd_atomic
            > attr.max_qp_init_rd_atomic || config.max_dest_rd_atomic > attr.max_qp_rd_atomic {
            return Err(RdmaError(22));
        }
        if pd.context.raw != cq.device.raw || psn > 0xffffff {
            return Err(RdmaError(22));
        }
        let (lid, mtu, global, gid) = ffi::port(pd.context.raw, port, gid_index)?;
        let (raw, number) = ffi::create_qp(pd.raw, cq.raw, config)?;
        Ok(
            Self {
                raw,
                pd,
                cq_id: Ghost(cq.id()),
                source: Tracked(source),
                endpoint: QueuePairEndpoint {
                    number,
                    psn,
                    lid,
                    gid,
                    mtu,
                    responder_resources: config.max_dest_rd_atomic,
                    origin: Ghost(source.node()),
                },
                port,
                gid_index,
                global,
                config,
            },
        )
    }

    pub fn endpoint(&self) -> (r: QueuePairEndpoint)
        ensures
            r.node() == self.source().node(),
            r.responder_resources_spec() == self.configuration().max_dest_rd_atomic,
    {
        QueuePairEndpoint {
            responder_resources: self.config.max_dest_rd_atomic,
            origin: Ghost(self.source().node()),
            ..self.endpoint
        }
    }
}

pub fn connect_rc<'a>(
    qp: UnconnectedQueuePair<'a>,
    peer: QueuePairEndpoint,
    _peer_node: NodeId,
) -> (r: Result<QueuePair<'a>, RdmaError>)
    requires
        qp.source().well_formed(),
        qp.source().source() is RdmaQueuePair,
        qp.source().node() != _peer_node,
        peer.node() == _peer_node,
    ensures
        r matches Ok(q) ==> q.ready() && q.source() == qp.source() && q.pd_id() == qp.pd_id()
            && q.cq_id() == qp.cq_id() && q.peer() == _peer_node && q.reads() == ISet::empty()
            && q.read_infos().dom() == ISet::empty() && q.configuration() == qp.configuration()
            && q.effective_max_rd_atomic() == (if qp.configuration().max_rd_atomic
            <= peer.responder_resources_spec() {
            qp.configuration().max_rd_atomic
        } else {
            peer.responder_resources_spec()
        }),
{
    proof {
        use_type_invariant(&qp);
    }
    if let Err(e) = ffi::connect(&qp, peer) {
        let _ = ffi::destroy_qp(qp.raw);
        return Err(e);
    }
    let eff = if qp.config.max_rd_atomic <= peer.responder_resources {
        qp.config.max_rd_atomic
    } else {
        peer.responder_resources
    };
    let tracked infos = SubmittedMap::empty(qp.source().id());
    Ok(
        QueuePair {
            connection: qp,
            effective_max_rd_atomic: eff,
            state: Ghost(QueuePairState::ReadyToSend),
            peer_node: Ghost(_peer_node),
            pending_reads: Ghost(ISet::empty()),
            pending_read_infos: Tracked(infos),
        },
    )
}

impl<'a> QueuePair<'a> {
    pub open spec fn id(&self) -> nat {
        self.source().source()->RdmaQueuePair_id
    }

    pub open(super) spec fn state(&self) -> QueuePairState {
        self.state@
    }

    pub open(super) spec fn configuration(&self) -> QueuePairConfig {
        self.connection.config
    }

    pub open spec fn max_send_sge(&self) -> u32 {
        self.configuration().max_send_sge
    }

    pub open spec fn send_depth(&self) -> u32 {
        self.configuration().send_depth
    }

    pub open(super) spec fn effective_max_rd_atomic(&self) -> u8 {
        self.effective_max_rd_atomic
    }

    pub fn effective_max_rd_atomic_exec(&self) -> (n: u8)
        ensures
            n == self.effective_max_rd_atomic(),
    {
        self.effective_max_rd_atomic
    }

    pub fn config(&self) -> (r: QueuePairConfig)
        ensures
            r == self.configuration(),
    {
        self.connection.config
    }

    pub fn reset_endpoint(&self, psn: u32) -> (r: QueuePairEndpoint)
        requires
            psn <= 0xffffff,
        ensures
            r.node() == self.source().node(),
            r.responder_resources_spec() == self.configuration().max_dest_rd_atomic,
    {
        QueuePairEndpoint {
            psn,
            responder_resources: self.connection.config.max_dest_rd_atomic,
            origin: Ghost(self.source().node()),
            ..self.connection.endpoint
        }
    }

    pub open(super) spec fn ready(&self) -> bool {
        self.max_send_sge() >= 1 && self.send_depth() >= 1 && self.state@ is ReadyToSend
            && self.source().well_formed() && self.source().source() is RdmaQueuePair
            && self.read_infos().instance_id() == self.source().id() && self.reads()
            == self.read_infos().dom()
    }

    pub open(super) spec fn source(&self) -> EventSourceToken {
        self.connection.source@
    }

    /// The next event this ready QP will submit is absent from any `SubmittedMap`.
    pub fn lemma_next_event_unsubmitted(
        &self,
        Tracked(instance): Tracked<&TransferInstance>,
        Tracked(submitted): Tracked<&SubmittedMap>,
    )
        requires
            self.ready(),
            self.source().id() == instance.id(),
            submitted.instance_id() == instance.id(),
        ensures
            !submitted.dom().contains(self.source().next_event()),
    {
        proof {
            self.connection.source.borrow().lemma_next_event_unsubmitted(instance, submitted);
        }
    }

    pub open(super) spec fn pd_id(&self) -> nat {
        self.connection.pd.id()
    }

    pub open(super) spec fn cq_id(&self) -> nat {
        self.connection.cq_id@
    }

    pub open(super) spec fn peer(&self) -> NodeId {
        self.peer_node@
    }

    pub open(super) spec fn reads(&self) -> ISet<TransferEvent> {
        self.pending_reads@
    }

    pub open(super) spec fn read_infos(&self) -> SubmittedMap {
        self.pending_read_infos@
    }

    pub open spec fn can_post(&self, cq: CompletionQueue, id: u64) -> bool {
        self.ready() && cq.well_formed() && cq.id() == self.cq_id() && cq.observer()
            == self.source().node() && cq.infos().instance_id() == self.source().id()
            && !cq.pending().contains_key(id) && cq.pending().len() < cq.capacity()
    }

    /// Poll this QP's signaled WRs before destroying it. Other QPs can keep using the CQ.
    pub fn destroy(self, _cq: &CompletionQueue) -> Result<(), RdmaError>
        requires
            _cq.id() == self.cq_id(),
            _cq.observer() == self.source().node(),
            forall|wr_id: u64|
                _cq.pending().contains_key(wr_id) ==> (#[trigger] _cq.pending()[wr_id]).kind.qp_id()
                    != self.id(),
    {
        ffi::destroy_qp(self.connection.raw)
    }
}

#[cfg_attr(not(verus_only), allow(dead_code))]
pub tracked struct QpErrorWitness {
    ghost qp_id: nat,
    ghost node: NodeId,
}

impl QpErrorWitness {
    pub closed spec fn matches_qp(self, qp_id: nat, node: NodeId) -> bool {
        self.qp_id == qp_id && self.node == node
    }

    pub proof fn duplicate(tracked &self) -> (tracked copy: Self)
        ensures
            copy == *self,
    {
        Self { qp_id: self.qp_id, node: self.node }
    }
}

pub fn modify_queue_pair_error(qp: &mut QueuePair) -> (r: Result<
    Tracked<QpErrorWitness>,
    RdmaError,
>)
    ensures
        r.is_err() ==> *final(qp) == *old(qp),
        r matches Ok(w) ==> final(qp).state() == QueuePairState::Error && final(qp).id() == old(
            qp,
        ).id() && final(qp).cq_id() == old(qp).cq_id() && final(qp).source() == old(qp).source()
            && final(qp).peer() == old(qp).peer() && final(qp).pd_id() == old(qp).pd_id()
            && final(qp).configuration() == old(qp).configuration() && w@.matches_qp(
            old(qp).id(),
            old(qp).source().node(),
        ),
{
    ffi::error(qp.connection.raw)?;
    proof {
        qp.state = Ghost(QueuePairState::Error);
    }
    Ok(Tracked(QpErrorWitness { qp_id: qp.id(), node: qp.source().node() }))
}

#[cfg_attr(not(verus_only), allow(dead_code))]
pub tracked struct QpResetWitness {
    ghost qp_id: nat,
    ghost node: NodeId,
}

impl QpResetWitness {
    pub closed spec fn matches_qp(self, qp_id: nat, node: NodeId) -> bool {
        self.qp_id == qp_id && self.node == node
    }

    pub proof fn duplicate(tracked &self) -> (tracked copy: Self)
        ensures
            copy == *self,
    {
        Self { qp_id: self.qp_id, node: self.node }
    }
}

pub fn reset_queue_pair(qp: &mut QueuePair, _cq: &CompletionQueue) -> (r: Result<
    Tracked<QpResetWitness>,
    RdmaError,
>)
    requires
        _cq.id() == old(qp).cq_id(),
        _cq.observer() == old(qp).source().node(),
        forall|wr_id: u64|
            _cq.pending().contains_key(wr_id) ==> (#[trigger] _cq.pending()[wr_id]).kind.qp_id()
                != old(qp).id(),
    ensures
        r.is_err() ==> *final(qp) == *old(qp),
        r matches Ok(w) ==> final(qp).state() == QueuePairState::Reset && final(qp).reads()
            == ISet::empty() && final(qp).read_infos().dom() == ISet::empty() && final(qp).cq_id()
            == old(qp).cq_id() && final(qp).source() == old(qp).source() && final(qp).pd_id()
            == old(qp).pd_id() && final(qp).peer() == old(qp).peer() && final(qp).configuration()
            == old(qp).configuration() && w@.matches_qp(old(qp).id(), old(qp).source().node()),
{
    ffi::reset(qp.connection.raw)?;
    proof {
        qp.pending_reads = Ghost(ISet::empty());
        qp.pending_read_infos = Tracked(SubmittedMap::empty(qp.source().id()));
        qp.state = Ghost(QueuePairState::Reset);
    }
    Ok(Tracked(QpResetWitness { qp_id: qp.id(), node: qp.source().node() }))
}

pub fn reconnect_rc(qp: &mut QueuePair, psn: u32, peer: QueuePairEndpoint) -> (r: Result<
    (),
    RdmaError,
>)
    requires
        old(qp).state() == QueuePairState::Reset,
        psn <= 0xffffff,
        peer.node() == old(qp).peer(),
        old(qp).source().well_formed(),
        old(qp).source().source() is RdmaQueuePair,
        old(qp).source().node() != old(qp).peer(),
    ensures
        r.is_err() ==> *final(qp) == *old(qp),
        r is Ok ==> final(qp).ready() && final(qp).state() == QueuePairState::ReadyToSend
            && final(qp).id() == old(qp).id() && final(qp).source() == old(qp).source()
            && final(qp).pd_id() == old(qp).pd_id() && final(qp).cq_id() == old(qp).cq_id()
            && final(qp).peer() == old(qp).peer() && final(qp).configuration() == old(
            qp,
        ).configuration() && final(qp).reads() == ISet::empty() && final(qp).read_infos().dom()
            == ISet::empty() && final(qp).effective_max_rd_atomic() == (if old(
            qp,
        ).configuration().max_rd_atomic <= peer.responder_resources_spec() {
            old(qp).configuration().max_rd_atomic
        } else {
            peer.responder_resources_spec()
        }),
{
    proof {
        use_type_invariant(&qp.connection);
    }
    ffi::reconnect(&qp.connection, psn, peer)?;
    let eff = if qp.connection.config.max_rd_atomic <= peer.responder_resources {
        qp.connection.config.max_rd_atomic
    } else {
        peer.responder_resources
    };
    qp.connection.endpoint.psn = psn;
    qp.effective_max_rd_atomic = eff;
    proof {
        qp.pending_reads = Ghost(ISet::empty());
        qp.pending_read_infos = Tracked(SubmittedMap::empty(qp.source().id()));
        qp.state = Ghost(QueuePairState::ReadyToSend);
    }
    Ok(())
}

} // verus!
