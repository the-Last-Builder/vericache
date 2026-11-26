//! Write-once persistent key-value map tokens.
//!
//! Each key `K` starts in the exclusive `unused` set and can be bound at most
//! once to a value `V`, producing a persistent `bindings` token. Any two
//! `bindings` tokens for the same instance and key agree on their value.
// The state machine macro names its generated module after the machine.
#![allow(non_snake_case)]

use verus_state_machines_macros::tokenized_state_machine;
use vstd::prelude::*;
#[cfg(verus_only)]
use vstd::tokens::{ElementToken, KeyValueToken};

verus! {

broadcast use vstd::iset_lib::group_iset_properties;

tokenized_state_machine!{ #[cfg_attr(verus_keep_ghost, verifier::reject_recursive_types(K))] WriteOnceTokens<K, V> {
    fields {
        /// Exclusive: each token is the right to bind one key.
        #[sharding(iset)]
        pub unused: ISet<K>,

        /// Persistent: each token records a key's immutable bound value.
        #[sharding(persistent_imap)]
        pub bindings: IMap<K, V>,
    }

    #[invariant]
    pub fn disjoint_domains(&self) -> bool {
        self.unused.disjoint(self.bindings.dom())
    }

    init!{
        initialize() {
            init unused = ISet::full();
            init bindings = IMap::empty();
        }
    }

    transition!{
        bind(k: K, v: V) {
            remove unused -= set { k };
            add bindings (union)= [k => v];
        }
    }

    property!{
        unused_key_has_no_binding(k: K, v: V) {
            have unused >= set { k };
            have bindings >= [k => v];
            assert(false);
        }
    }

    #[inductive(initialize)]
    fn initialize_inductive(post: Self) {}

    #[inductive(bind)]
    fn bind_inductive(pre: Self, post: Self, k: K, v: V) {}
}}

/// Checked trait wrapper exposing `WriteOnceTokens::Instance` transitions to
/// downstream crates during verification.
pub trait WriteOnceTransitions<K, V>: Sized {
    spec fn write_once_instance_id(&self) -> vstd::tokens::InstanceId;

    proof fn initialize() -> (tracked result: (
        Tracked<Self>,
        Tracked<WriteOnceTokens::unused_set<K, V>>,
    ))
        ensures
            result.1@.instance_id() == result.0@.write_once_instance_id(),
            result.1@.set() == ISet::<K>::full(),
    ;

    proof fn bind(
        tracked &self,
        k: K,
        v: V,
        tracked unused_key: WriteOnceTokens::unused<K, V>,
    ) -> (tracked result: WriteOnceTokens::bindings<K, V>)
        requires
            unused_key.instance_id() == self.write_once_instance_id(),
            unused_key.element() == k,
        ensures
            result.instance_id() == self.write_once_instance_id(),
            result.key() == k,
            result.value() == v,
    ;
}

impl<K, V> WriteOnceTransitions<K, V> for WriteOnceTokens::Instance<K, V> {
    open spec fn write_once_instance_id(&self) -> vstd::tokens::InstanceId {
        self.id()
    }

    proof fn initialize() -> (tracked result: (
        Tracked<Self>,
        Tracked<WriteOnceTokens::unused_set<K, V>>,
    )) {
        let tracked (Tracked(inst), Tracked(unused), _) = WriteOnceTokens::Instance::<
            K,
            V,
        >::initialize();
        (Tracked(inst), Tracked(unused))
    }

    proof fn bind(
        tracked &self,
        k: K,
        v: V,
        tracked unused_key: WriteOnceTokens::unused<K, V>,
    ) -> (tracked result: WriteOnceTokens::bindings<K, V>) {
        WriteOnceTokens::Instance::bind(self, k, v, unused_key)
    }
}

/// Any two `bindings` tokens for the same instance and key agree on the value.
pub proof fn binding_agree<K, V>(
    tracked a: &WriteOnceTokens::bindings<K, V>,
    tracked b: &WriteOnceTokens::bindings<K, V>,
)
    requires
        a.instance_id() == b.instance_id(),
        a.key() == b.key(),
    ensures
        a.value() == b.value(),
{
    a.agree(b);
}

/// Persistent `bindings` tokens can be duplicated freely.
pub proof fn binding_duplicate<K, V>(tracked tok: &WriteOnceTokens::bindings<K, V>) -> (tracked dup:
    WriteOnceTokens::bindings<K, V>)
    ensures
        dup == *tok,
{
    tok.clone()
}

/// Binding and unused-key tokens from the same instance describe distinct keys.
pub proof fn unused_key_differs_from_binding<K, V>(
    tracked instance: &WriteOnceTokens::Instance<K, V>,
    tracked unused: &WriteOnceTokens::unused<K, V>,
    tracked binding: &WriteOnceTokens::bindings<K, V>,
)
    requires
        unused.instance_id() == instance.id(),
        binding.instance_id() == instance.id(),
    ensures
        unused.element() != binding.key(),
{
    if unused.element() == binding.key() {
        instance.unused_key_has_no_binding(unused.element(), binding.value(), unused, binding);
    }
}

} // verus!
