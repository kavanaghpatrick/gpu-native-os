//! AST definitions for GPU DSL
//!
//! THE GPU IS THE COMPUTER.
//! This AST represents GPU-native constructs.

/// Type system for GPU values
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Type {
    F32,
    I32,
    U32,
    F32x4,
    Bool,
    Void,
}

impl Type {
    pub fn is_integer(&self) -> bool {
        matches!(self, Type::I32 | Type::U32)
    }

    pub fn is_float(&self) -> bool {
        matches!(self, Type::F32 | Type::F32x4)
    }
}

/// A GPU kernel function
#[derive(Debug)]
pub struct GpuFunction {
    pub name: String,
    pub params: Vec<(String, Type)>,
    pub body: Vec<Statement>,
}

/// Statements in the GPU DSL
#[derive(Debug)]
pub enum Statement {
    /// Variable declaration: let x = expr;
    Let {
        name: String,
        ty: Option<Type>,
        value: Expr,
    },
    /// Assignment: x = expr; or STATE[i] = expr;
    Assign {
        target: AssignTarget,
        value: Expr,
    },
    /// If statement
    If {
        condition: Expr,
        then_body: Vec<Statement>,
        else_body: Option<Vec<Statement>>,
    },
    /// For loop: for i in start..end { }
    For {
        var: String,
        start: Expr,
        end: Expr,
        body: Vec<Statement>,
    },
    /// While loop: while cond { }
    While {
        condition: Expr,
        body: Vec<Statement>,
    },
    /// Break out of loop
    Break,
    /// Continue to next iteration
    Continue,
    /// Expression statement (for side effects like function calls)
    Expr(Expr),
}

/// Assignment targets
#[derive(Debug)]
pub enum AssignTarget {
    /// Simple variable: x = ...
    Variable(String),
    /// State memory: STATE[idx] = ...
    State(Expr),
    /// Atomic memory: ATOMIC[idx] = ...
    Atomic(Expr),
}

/// Expressions in the GPU DSL
#[derive(Debug)]
pub enum Expr {
    /// Literal values
    LitInt(i64),
    LitFloat(f64),
    LitBool(bool),

    /// Variable reference
    Var(String),

    /// Binary operation
    Binary {
        op: BinOp,
        left: Box<Expr>,
        right: Box<Expr>,
    },

    /// Unary operation
    Unary {
        op: UnaryOp,
        operand: Box<Expr>,
    },

    /// Function call: func(args...)
    Call {
        func: String,
        args: Vec<Expr>,
    },

    /// Field access: expr.field
    Field {
        base: Box<Expr>,
        field: String,
    },

    /// Type cast: expr as Type
    Cast {
        expr: Box<Expr>,
        ty: Type,
    },

    /// State memory read: STATE[idx]
    StateRead(Box<Expr>),

    /// Atomic memory read: ATOMIC[idx]
    AtomicRead(Box<Expr>),
}

/// Binary operators
#[derive(Debug, Clone, Copy)]
pub enum BinOp {
    // Arithmetic
    Add,
    Sub,
    Mul,
    Div,
    Rem,

    // Comparison
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,

    // Logical
    And,
    Or,

    // Bitwise
    BitAnd,
    BitOr,
    BitXor,
    Shl,
    Shr,
}

/// Unary operators
#[derive(Debug, Clone, Copy)]
pub enum UnaryOp {
    Neg,
    Not,
    BitNot,
}
