#![feature(rustc_private)]
#![warn(unused_extern_crates)]

//! Soroban cost-analysis lints.
//!
//! This crate is a [Dylint](https://github.com/trailofbits/dylint) library. It
//! is compiled to a `cdylib` and loaded by `cargo dylint` (driven by the
//! `cargo-cost-lint` wrapper), which runs each lint as a late-stage pass over a
//! Soroban contract's [HIR](https://rustc-dev-guide.rust-lang.org/hir.html).
//!
//! # What the lints look for
//!
//! Soroban meters execution against a CPU and memory budget. The lints here
//! flag *structural* anti-patterns whose cost does not depend on runtime input,
//! so they can be caught statically:
//!
//! - [`SOROBAN_STORAGE_IN_LOOP`] — storage reads/writes performed inside a loop.
//! - [`REDUNDANT_ENV_CLONE`] — cloning the `Env` handle when a reference would
//!   do.
//! - [`UNNECESSARY_HOST_FUNCTION_CALL`] — a metered host call inside a loop
//!   whose result is invariant across iterations and could be hoisted out.
//! - [`HOST_IN_LOOP`] — use of a `Host` object inside a loop.
//! - [`SYMBOL_NEW_FOR_SHORT_LITERAL`] — `Symbol::new` on a literal short enough
//!   for the compile-time `symbol_short!` macro.
//!
//! Each lint is assigned a [`LintCategory`] and registered in [`LINT_METADATA`],
//! the single source of truth the wrapper reads to describe available lints.
//!
//! # How a lint is structured
//!
//! Every lint follows the same three-part shape used throughout `rustc`/Clippy:
//!
//! 1. A [`declare_lint!`](rustc_session::declare_lint) invocation that defines
//!    the lint's static descriptor, default level, and short description.
//! 2. A zero-sized marker struct (e.g. [`SorobanStorageInLoop`]) that the pass
//!    is dispatched on.
//! 3. An `impl` of [`LateLintPass`] for that struct whose `check_expr` inspects
//!    each expression and emits a diagnostic when the pattern matches.
//!
//! Type-based matching is done against `soroban_sdk` def-paths via
//! [`match_soroban_def_path`] and the `SOROBAN_*` path tables, so the lints key
//! off the SDK's public types rather than fragile name heuristics.
//!
//! # Adding a lint
//!
//! See `CONTRIBUTING.md`. In short: declare the lint, add a marker struct and
//! `LateLintPass` impl, register both in [`register_lints`], and add a
//! [`LintMetadata`] entry to [`LINT_METADATA`] with the appropriate
//! [`LintCategory`].

extern crate rustc_ast;
extern crate rustc_errors;
extern crate rustc_hir;
extern crate rustc_lint;
extern crate rustc_middle;
extern crate rustc_session;
extern crate rustc_span;

use clippy_utils::diagnostics::{span_lint_and_help, span_lint_and_sugg};
use clippy_utils::get_enclosing_loop_or_multi_call_closure;
use clippy_utils::source::snippet_opt;
use clippy_utils::usage::mutated_variables;
use rustc_ast::LitKind;
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_hir::intravisit::{self, Visitor};
use rustc_hir::{HirId, HirIdSet};
use rustc_lint::{LateContext, LateLintPass, LintStore};
use rustc_span::def_id::DefId;

dylint_linting::dylint_library!();

fn match_soroban_def_path<'tcx>(cx: &LateContext<'tcx>, def_id: DefId, segments: &[&str]) -> bool {
    let full = cx.tcx.def_path_str(def_id);
    let suffix: String = segments.join("::");
    full.ends_with(&suffix)
}

/// Soroban storage accessor types. Every method call on one of these reaches
/// the host's storage subsystem.
const SOROBAN_STORAGE_TYPES: &[&[&str]] = &[
    &["soroban_sdk", "storage", "Storage"],
    &["soroban_sdk", "storage", "Instance"],
    &["soroban_sdk", "storage", "Persistent"],
    &["soroban_sdk", "storage", "Temporary"],
];

/// Soroban host accessor types reachable from `Env`. A method call on any of
/// them crosses the guest/host boundary and is metered, so repeating it inside
/// a loop with unchanged inputs is wasted CPU budget.
///
/// `soroban_sdk::storage::*` is deliberately absent: storage operations in a
/// loop are reported by [`SOROBAN_STORAGE_IN_LOOP`] instead.
const SOROBAN_HOST_TYPES: &[&[&str]] = &[
    &["soroban_sdk", "ledger", "Ledger"],
    &["soroban_sdk", "crypto", "Crypto"],
    &["soroban_sdk", "crypto", "CryptoHazmat"],
    &["soroban_sdk", "crypto", "bls12_381", "Bls12_381"],
    &["soroban_sdk", "crypto", "bn254", "Bn254"],
    &["soroban_sdk", "prng", "Prng"],
    &["soroban_sdk", "events", "Events"],
    &["soroban_sdk", "deploy", "Deployer"],
    &["soroban_sdk", "deploy", "DeployerWithAddress"],
    &["soroban_sdk", "deploy", "DeployerWithAsset"],
];

/// Host calls that live directly on `Env` rather than on an accessor type, and
/// whose result is constant for the whole invocation.
///
/// The accessor methods themselves (`Env::ledger`, `Env::crypto`, ...) are not
/// listed: they only build a wrapper value, the metered work happens in the
/// method called on the wrapper. Argument-taking `Env` methods such as
/// `invoke_contract` or `authorize_as_current_contract` are also excluded
/// because their cost is inherent to what the loop is doing.
const SOROBAN_ENV_HOST_METHODS: &[&str] = &["current_contract_address"];

fn matches_any_path<'tcx>(cx: &LateContext<'tcx>, def_id: DefId, paths: &[&[&str]]) -> bool {
    paths
        .iter()
        .any(|segments| match_soroban_def_path(cx, def_id, segments))
}

/// Collects the `HirId`s of every binding introduced inside the visited
/// subtree, e.g. the loop variable of a `for` loop or a per-iteration `let`.
#[derive(Default)]
struct BindingCollector {
    bindings: HirIdSet,
}

impl<'tcx> Visitor<'tcx> for BindingCollector {
    /// Records the `HirId` of any binding pattern encountered, then recurses
    /// into sub-patterns so nested bindings (e.g. `(a, b)`) are all captured.
    fn visit_pat(&mut self, pat: &'tcx hir::Pat<'tcx>) {
        if let hir::PatKind::Binding(_, hir_id, _, _) = pat.kind {
            self.bindings.insert(hir_id);
        }
        intravisit::walk_pat(self, pat);
    }
}

/// Collects the `HirId`s of every local read in the visited subtree.
#[derive(Default)]
struct LocalReadCollector {
    reads: Vec<HirId>,
}

impl<'tcx> Visitor<'tcx> for LocalReadCollector {
    /// Records the `HirId` of every resolved read of a local variable, i.e. a
    /// path expression that resolves to a `Res::Local`, then recurses into the
    /// rest of the expression tree.
    fn visit_expr(&mut self, expr: &'tcx hir::Expr<'tcx>) {
        if let hir::ExprKind::Path(hir::QPath::Resolved(None, path)) = expr.kind
            && let hir::def::Res::Local(hir_id) = path.res
        {
            self.reads.push(hir_id);
        }
        intravisit::walk_expr(self, expr);
    }
}

/// Whether `call` — receiver chain and arguments included — reads anything that
/// changes from iteration to iteration of `loop_expr`.
///
/// Such a call is doing real per-iteration work, so hoisting it out of the loop
/// would change behaviour and it must not be reported. The answer errs towards
/// "depends": when the mutation analysis cannot give a verdict, the call is
/// treated as loop-dependent and stays unreported.
///
/// Known gaps, all of which cause a call to be reported rather than skipped:
/// bindings and mutations inside a closure body nested in the loop are not
/// seen, and mutation through a raw pointer or interior mutability (`RefCell`,
/// `Cell`) is not tracked.
fn depends_on_loop_state<'tcx>(
    cx: &LateContext<'tcx>,
    loop_expr: &'tcx hir::Expr<'tcx>,
    call: &'tcx hir::Expr<'tcx>,
) -> bool {
    let Some(mutated) = mutated_variables(loop_expr, cx) else {
        return true;
    };

    let mut bound = BindingCollector::default();
    bound.visit_expr(loop_expr);

    let mut read = LocalReadCollector::default();
    read.visit_expr(call);

    read.reads
        .iter()
        .any(|hir_id| bound.bindings.contains(hir_id) || mutated.contains(hir_id))
}

/// Whether `expr` sits directly inside a loop body, returning that loop.
///
/// A call inside a closure that the loop calls is not reported: the closure may
/// well be defined elsewhere, and the receiver is out of reach for the
/// loop-dependence analysis.
fn enclosing_loop<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &'tcx hir::Expr<'tcx>,
) -> Option<&'tcx hir::Expr<'tcx>> {
    let enclosing = get_enclosing_loop_or_multi_call_closure(cx, expr)?;
    matches!(enclosing.kind, hir::ExprKind::Loop(..)).then_some(enclosing)
}

/// The cost dimension a lint speaks to.
///
/// Every lint is tagged with exactly one category so the `cargo-cost-lint`
/// wrapper can group findings by the kind of budget they affect. The mapping
/// lives in [`LINT_METADATA`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LintCategory {
    /// Reads and writes to Soroban storage (instance, persistent, temporary).
    StorageOperations,
    /// CPU-metered work, such as redundant host calls.
    Compute,
    /// Memory allocation and copying, such as needless clones.
    Memory,
    /// Creation and expiry of storage entries.
    EntryLifecycle,
    /// Construction and handling of `Symbol` values.
    SymbolOperations,
}

/// A single entry in the lint registry: a lint paired with its [`LintCategory`].
///
/// See [`LINT_METADATA`] for the full table.
pub struct LintMetadata {
    /// The lint this entry describes.
    pub lint: &'static rustc_lint::Lint,
    /// The cost dimension the lint is grouped under.
    pub category: LintCategory,
}

/// The registry of every lint shipped by this crate, each paired with its
/// [`LintCategory`].
///
/// This is the single source of truth for lint metadata: the `cargo-cost-lint`
/// wrapper reads it to enumerate and categorize available lints. Any lint added
/// in [`register_lints`] should also gain an entry here.
pub const LINT_METADATA: &[LintMetadata] = &[
    LintMetadata {
        lint: SOROBAN_STORAGE_IN_LOOP,
        category: LintCategory::StorageOperations,
    },
    LintMetadata {
        lint: REDUNDANT_ENV_CLONE,
        category: LintCategory::Memory,
    },
    LintMetadata {
        lint: UNNECESSARY_HOST_FUNCTION_CALL,
        category: LintCategory::Compute,
    },
    LintMetadata {
        lint: HOST_IN_LOOP,
        category: LintCategory::Compute,
    },
    LintMetadata {
        lint: SYMBOL_NEW_FOR_SHORT_LITERAL,
        category: LintCategory::SymbolOperations,
    },
];

/// Dylint entry point: registers every lint and its late pass with the
/// compiler's [`LintStore`].
///
/// `cargo dylint` calls this once per crate being checked. The set of lints
/// registered here must stay in sync with [`LINT_METADATA`]. The session
/// argument is unused; lint registration does not depend on session state.
#[unsafe(no_mangle)]
pub fn register_lints(_sess: &rustc_session::Session, lint_store: &mut LintStore) {
    lint_store.register_lints(&[
        SOROBAN_STORAGE_IN_LOOP,
        REDUNDANT_ENV_CLONE,
        UNNECESSARY_HOST_FUNCTION_CALL,
        HOST_IN_LOOP,
        SYMBOL_NEW_FOR_SHORT_LITERAL,
    ]);
    lint_store.register_late_pass(|_| Box::new(SorobanStorageInLoop));
    lint_store.register_late_pass(|_| Box::new(RedundantEnvClone));
    lint_store.register_late_pass(|_| Box::new(UnnecessaryHostFunctionCall));
    lint_store.register_late_pass(|_| Box::new(HostInLoop));
    lint_store.register_late_pass(|_| Box::new(SymbolNewForShortLiteral));
}

rustc_session::declare_lint! {
    pub SOROBAN_STORAGE_IN_LOOP,
    Warn,
    "storage operations inside a loop"
}
/// Late pass backing [`SOROBAN_STORAGE_IN_LOOP`].
pub struct SorobanStorageInLoop;
rustc_session::impl_lint_pass!(SorobanStorageInLoop => [SOROBAN_STORAGE_IN_LOOP]);

impl<'tcx> LateLintPass<'tcx> for SorobanStorageInLoop {
    /// Flags a method call whose receiver is a Soroban storage accessor (or
    /// `Env::storage`) when it sits inside a loop.
    ///
    /// Storage access is metered on every iteration, so performing it in a loop
    /// multiplies the cost. The receiver type is matched against
    /// [`SOROBAN_STORAGE_TYPES`]; the loop check uses [`enclosing_loop`]. No
    /// suggestion is offered because the fix (hoisting or batching) is
    /// context-specific, so only a help note is emitted.
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx hir::Expr<'tcx>) {
        if let hir::ExprKind::MethodCall(path_segment, receiver, _args, _span) = expr.kind {
            let receiver_ty = cx.typeck_results().expr_ty(receiver);
            let peeled_ty = receiver_ty.peel_refs();

            let is_storage_access = if let rustc_middle::ty::Adt(adt_def, _) = peeled_ty.kind() {
                let did = adt_def.did();
                matches_any_path(cx, did, SOROBAN_STORAGE_TYPES)
                    || (match_soroban_def_path(cx, did, &["soroban_sdk", "Env"])
                        && path_segment.ident.name.as_str() == "storage")
            } else {
                false
            };

            if is_storage_access && enclosing_loop(cx, expr).is_some() {
                span_lint_and_help(
                    cx,
                    SOROBAN_STORAGE_IN_LOOP,
                    expr.span,
                    "storage operation inside a loop",
                    None,
                    "move storage operations out of the loop or accumulate mutations in memory first",
                );
            }
        }
    }
}

rustc_session::declare_lint! {
    pub REDUNDANT_ENV_CLONE,
    Warn,
    "redundant clone on Env object"
}
/// Late pass backing [`REDUNDANT_ENV_CLONE`].
pub struct RedundantEnvClone;
rustc_session::impl_lint_pass!(RedundantEnvClone => [REDUNDANT_ENV_CLONE]);

impl<'tcx> LateLintPass<'tcx> for RedundantEnvClone {
    /// Flags a `.clone()` call whose receiver is a `soroban_sdk::Env`.
    ///
    /// `Env` is a cheap handle to the host and is almost always better passed
    /// by reference or value than cloned; the clone adds needless work. Matches
    /// the `clone` method name and confirms the receiver type resolves to
    /// `soroban_sdk::Env` before emitting a help note.
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx hir::Expr<'tcx>) {
        if let hir::ExprKind::MethodCall(path_segment, receiver, _args, _span) = expr.kind
            && path_segment.ident.name.as_str() == "clone"
        {
            let receiver_ty = cx.typeck_results().expr_ty(receiver);
            let peeled_ty = receiver_ty.peel_refs();

            let is_env = if let rustc_middle::ty::Adt(adt_def, _) = peeled_ty.kind() {
                match_soroban_def_path(cx, adt_def.did(), &["soroban_sdk", "Env"])
            } else {
                false
            };

            if is_env {
                span_lint_and_help(
                    cx,
                    REDUNDANT_ENV_CLONE,
                    expr.span,
                    "redundant clone on Env object",
                    None,
                    "pass Env by reference or value instead of cloning",
                );
            }
        }
    }
}

rustc_session::declare_lint! {
    pub UNNECESSARY_HOST_FUNCTION_CALL,
    Warn,
    "unnecessary host function call inside loop"
}
/// Late pass backing [`UNNECESSARY_HOST_FUNCTION_CALL`].
pub struct UnnecessaryHostFunctionCall;
rustc_session::impl_lint_pass!(UnnecessaryHostFunctionCall => [UNNECESSARY_HOST_FUNCTION_CALL]);

rustc_session::declare_lint! {
    pub HOST_IN_LOOP,
    Warn,
    "use of Host object inside a loop"
}
/// Late pass backing [`HOST_IN_LOOP`].
pub struct HostInLoop;
rustc_session::impl_lint_pass!(HostInLoop => [HOST_IN_LOOP]);

impl<'tcx> LateLintPass<'tcx> for UnnecessaryHostFunctionCall {
    /// Flags a metered host call inside a loop whose result is invariant across
    /// iterations, so it could be computed once and reused.
    ///
    /// The receiver must resolve to one of [`SOROBAN_HOST_TYPES`], or the call
    /// must be one of the constant-result `Env` methods in
    /// [`SOROBAN_ENV_HOST_METHODS`]. The call is only reported when it is inside
    /// a loop ([`enclosing_loop`]) *and* does not read loop-varying state
    /// ([`depends_on_loop_state`]); the latter guard keeps calls whose inputs
    /// change each iteration from being flagged.
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx hir::Expr<'tcx>) {
        if let hir::ExprKind::MethodCall(path_segment, receiver, _args, _span) = expr.kind {
            let receiver_ty = cx.typeck_results().expr_ty(receiver);
            let peeled_ty = receiver_ty.peel_refs();

            let is_host_function = if let rustc_middle::ty::Adt(adt_def, _) = peeled_ty.kind() {
                let did = adt_def.did();
                matches_any_path(cx, did, SOROBAN_HOST_TYPES)
                    || (match_soroban_def_path(cx, did, &["soroban_sdk", "Env"])
                        && SOROBAN_ENV_HOST_METHODS.contains(&path_segment.ident.name.as_str()))
            } else {
                false
            };

            if is_host_function
                && let Some(loop_expr) = enclosing_loop(cx, expr)
                && !depends_on_loop_state(cx, loop_expr, expr)
            {
                span_lint_and_help(
                    cx,
                    UNNECESSARY_HOST_FUNCTION_CALL,
                    expr.span,
                    "unnecessary host function call inside loop",
                    None,
                    "call this function outside the loop and reuse the result",
                );
            }
        }
    }
}

impl<'tcx> LateLintPass<'tcx> for HostInLoop {
    /// Flags any method call whose receiver is a `host::Host` object inside a
    /// loop.
    ///
    /// Unlike [`UnnecessaryHostFunctionCall`], this pass does not attempt a
    /// loop-invariance analysis: any `Host` use in a loop is surfaced with a
    /// help note suggesting the call be moved out where possible.
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx hir::Expr<'tcx>) {
        if let hir::ExprKind::MethodCall(_path_segment, receiver, _args, _span) = expr.kind {
            let receiver_ty = cx.typeck_results().expr_ty(receiver);
            let peeled_ty = receiver_ty.peel_refs();

            let is_host = if let rustc_middle::ty::Adt(adt_def, _) = peeled_ty.kind() {
                match_soroban_def_path(cx, adt_def.did(), &["host", "Host"])
            } else {
                false
            };

            if is_host && enclosing_loop(cx, expr).is_some() {
                span_lint_and_help(
                    cx,
                    HOST_IN_LOOP,
                    expr.span,
                    "use of Host object inside a loop",
                    None,
                    "consider moving the Host usage outside the loop if possible",
                );
            }
        }
    }
}

// =======================================================================
// symbol_new_for_short_literal — Lint
// =======================================================================

rustc_session::declare_lint! {
    pub SYMBOL_NEW_FOR_SHORT_LITERAL,
    Warn,
    "Symbol::new used with a short literal that could use symbol_short! macro"
}
/// Late pass backing [`SYMBOL_NEW_FOR_SHORT_LITERAL`].
pub struct SymbolNewForShortLiteral;
rustc_session::impl_lint_pass!(SymbolNewForShortLiteral => [SYMBOL_NEW_FOR_SHORT_LITERAL]);

impl<'tcx> LateLintPass<'tcx> for SymbolNewForShortLiteral {
    /// Flags `Symbol::new(&env, "literal")` when the literal is short enough to
    /// build at compile time with the `symbol_short!` macro.
    ///
    /// `Symbol::new` constructs the symbol at runtime, which is metered;
    /// `symbol_short!` produces it as a compile-time constant instead. The pass
    /// recognizes a two-argument call to `soroban_sdk::Symbol::new` whose second
    /// argument is a string literal accepted by [`is_valid_short_symbol`]. When
    /// the argument snippet is available it emits a machine-applicable
    /// suggestion; otherwise it falls back to a help note.
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx hir::Expr<'tcx>) {
        // Check for Symbol::new(&env, "literal") calls
        if let hir::ExprKind::Call(callee, args) = expr.kind
            && args.len() == 2
            && let hir::ExprKind::Path(ref qpath) = callee.kind
            && let Some(def_id) = cx.qpath_res(qpath, callee.hir_id).opt_def_id()
            && match_soroban_def_path(cx, def_id, &["soroban_sdk", "Symbol", "new"])
        {
            // Check if the second argument is a string literal
            if let hir::ExprKind::Lit(lit) = args[1].kind
                && let LitKind::Str(symbol, _) = lit.node
            {
                let s = symbol.as_str();
                if is_valid_short_symbol(s) {
                    // Check if there's a valid suggestion
                    if let Some(snippet) = snippet_opt(cx, args[1].span) {
                        let suggestion = format!("symbol_short!({})", snippet);
                        span_lint_and_sugg(
                            cx,
                            SYMBOL_NEW_FOR_SHORT_LITERAL,
                            expr.span,
                            "Symbol::new called with a short literal that could use symbol_short! macro",
                            "use symbol_short! macro for compile-time symbol creation",
                            suggestion,
                            Applicability::MachineApplicable,
                        );
                    } else {
                        span_lint_and_help(
                            cx,
                            SYMBOL_NEW_FOR_SHORT_LITERAL,
                            expr.span,
                            "Symbol::new called with a short literal that could use symbol_short! macro",
                            None,
                            "use symbol_short! macro for compile-time symbol creation",
                        );
                    }
                }
            }
        }
    }
}

/// Check if a string is a valid short symbol (<= 9 chars, only a-zA-Z0-9_)
fn is_valid_short_symbol(s: &str) -> bool {
    if s.len() > 9 || s.is_empty() {
        return false;
    }
    s.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
}

#[test]
fn ui() {
    dylint_testing::ui_test(env!("CARGO_PKG_NAME"), "ui");
}
