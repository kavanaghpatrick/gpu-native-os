//! Parser for GPU DSL - converts syn AST to our GPU AST
//!
//! THE GPU IS THE COMPUTER.
//! Parse Rust-like syntax into GPU-native constructs.

use crate::ast::*;
use proc_macro2::Span;
use syn::{
    parse::{Parse, ParseStream},
    punctuated::Punctuated,
    spanned::Spanned,
    token, Block, Expr as SynExpr, ExprAssign, ExprBinary, ExprBlock, ExprBreak, ExprCall,
    ExprCast, ExprContinue, ExprField, ExprIf, ExprIndex, ExprLit, ExprParen, ExprPath,
    ExprRange, ExprUnary, ExprWhile, Ident, ItemFn, Lit, Local, Pat, PatIdent, RangeLimits,
    Stmt, Token, Type as SynType,
};

/// Parse a GPU kernel function
impl Parse for GpuFunction {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let func: ItemFn = input.parse()?;

        let name = func.sig.ident.to_string();

        // Parse parameters
        let params = func
            .sig
            .inputs
            .iter()
            .filter_map(|arg| {
                if let syn::FnArg::Typed(pat_type) = arg {
                    if let Pat::Ident(PatIdent { ident, .. }) = &*pat_type.pat {
                        let ty = parse_type(&pat_type.ty)?;
                        return Some(Ok((ident.to_string(), ty)));
                    }
                }
                None
            })
            .collect::<syn::Result<Vec<_>>>()?;

        // Parse body
        let body = parse_block(&func.block)?;

        Ok(GpuFunction { name, params, body })
    }
}

fn parse_type(ty: &SynType) -> Option<Type> {
    match ty {
        SynType::Path(path) => {
            let ident = path.path.get_ident()?.to_string();
            match ident.as_str() {
                "f32" => Some(Type::F32),
                "i32" => Some(Type::I32),
                "u32" => Some(Type::U32),
                "bool" => Some(Type::Bool),
                _ => None,
            }
        }
        _ => None,
    }
}

fn parse_block(block: &Block) -> syn::Result<Vec<Statement>> {
    block.stmts.iter().map(parse_stmt).collect()
}

fn parse_stmt(stmt: &Stmt) -> syn::Result<Statement> {
    match stmt {
        Stmt::Local(local) => parse_local(local),
        Stmt::Expr(expr, _) => parse_expr_stmt(expr),
        Stmt::Item(_) => Err(syn::Error::new(
            stmt.span(),
            "Items not supported in GPU kernel",
        )),
        Stmt::Macro(m) => Err(syn::Error::new(m.span(), "Macros not supported in GPU kernel")),
    }
}

fn parse_local(local: &Local) -> syn::Result<Statement> {
    let name = match &local.pat {
        Pat::Ident(PatIdent { ident, .. }) => ident.to_string(),
        Pat::Wild(_) => "_".to_string(),
        _ => {
            return Err(syn::Error::new(
                local.pat.span(),
                "Only simple identifiers supported in let",
            ))
        }
    };

    let value = match &local.init {
        Some(init) => parse_expr(&init.expr)?,
        None => {
            return Err(syn::Error::new(
                local.span(),
                "Let binding must have initializer",
            ))
        }
    };

    Ok(Statement::Let {
        name,
        ty: None,
        value,
    })
}

fn parse_expr_stmt(expr: &SynExpr) -> syn::Result<Statement> {
    // Check for assignment
    if let SynExpr::Assign(ExprAssign { left, right, .. }) = expr {
        let target = parse_assign_target(left)?;
        let value = parse_expr(right)?;
        return Ok(Statement::Assign { target, value });
    }

    // Check for if
    if let SynExpr::If(expr_if) = expr {
        return parse_if(expr_if);
    }

    // Check for while
    if let SynExpr::While(expr_while) = expr {
        return parse_while(expr_while);
    }

    // Check for for loop
    if let SynExpr::ForLoop(expr_for) = expr {
        return parse_for(expr_for);
    }

    // Check for break
    if let SynExpr::Break(_) = expr {
        return Ok(Statement::Break);
    }

    // Check for continue
    if let SynExpr::Continue(_) = expr {
        return Ok(Statement::Continue);
    }

    // Otherwise, it's an expression statement
    Ok(Statement::Expr(parse_expr(expr)?))
}

fn parse_assign_target(expr: &SynExpr) -> syn::Result<AssignTarget> {
    match expr {
        SynExpr::Path(path) => {
            let name = path
                .path
                .get_ident()
                .ok_or_else(|| syn::Error::new(path.span(), "Expected identifier"))?
                .to_string();
            Ok(AssignTarget::Variable(name))
        }
        SynExpr::Index(ExprIndex { expr, index, .. }) => {
            if let SynExpr::Path(path) = &**expr {
                let base_name = path
                    .path
                    .get_ident()
                    .ok_or_else(|| syn::Error::new(path.span(), "Expected identifier"))?
                    .to_string();
                let idx = parse_expr(index)?;
                match base_name.as_str() {
                    "STATE" => Ok(AssignTarget::State(idx)),
                    "ATOMIC" => Ok(AssignTarget::Atomic(idx)),
                    _ => Err(syn::Error::new(
                        expr.span(),
                        "Only STATE[i] and ATOMIC[i] indexing supported",
                    )),
                }
            } else {
                Err(syn::Error::new(
                    expr.span(),
                    "Only STATE[i] and ATOMIC[i] indexing supported",
                ))
            }
        }
        _ => Err(syn::Error::new(
            expr.span(),
            "Invalid assignment target",
        )),
    }
}

fn parse_if(expr_if: &syn::ExprIf) -> syn::Result<Statement> {
    let condition = parse_expr(&expr_if.cond)?;
    let then_body = parse_block(&expr_if.then_branch)?;

    let else_body = match &expr_if.else_branch {
        Some((_, else_expr)) => {
            if let SynExpr::Block(ExprBlock { block, .. }) = &**else_expr {
                Some(parse_block(block)?)
            } else if let SynExpr::If(nested_if) = &**else_expr {
                // else if becomes else { if ... }
                Some(vec![parse_if(nested_if)?])
            } else {
                return Err(syn::Error::new(
                    else_expr.span(),
                    "Invalid else branch",
                ));
            }
        }
        None => None,
    };

    Ok(Statement::If {
        condition,
        then_body,
        else_body,
    })
}

fn parse_while(expr_while: &syn::ExprWhile) -> syn::Result<Statement> {
    let condition = parse_expr(&expr_while.cond)?;
    let body = parse_block(&expr_while.body)?;

    Ok(Statement::While { condition, body })
}

fn parse_for(expr_for: &syn::ExprForLoop) -> syn::Result<Statement> {
    let var = match &*expr_for.pat {
        Pat::Ident(PatIdent { ident, .. }) => ident.to_string(),
        Pat::Wild(_) => "_".to_string(),
        _ => {
            return Err(syn::Error::new(
                expr_for.pat.span(),
                "Only simple identifiers in for loop",
            ))
        }
    };

    // Parse range expression
    let (start, end) = match &*expr_for.expr {
        SynExpr::Range(ExprRange {
            start, end, limits, ..
        }) => {
            let s = match start {
                Some(e) => parse_expr(e)?,
                None => Expr::LitInt(0),
            };
            let e = match end {
                Some(e) => parse_expr(e)?,
                None => {
                    return Err(syn::Error::new(
                        expr_for.expr.span(),
                        "For loop range must have end",
                    ))
                }
            };
            // Note: Rust ranges are exclusive by default (start..end)
            // We'll handle both .. and ..= in codegen
            (s, e)
        }
        _ => {
            return Err(syn::Error::new(
                expr_for.expr.span(),
                "For loop must use range syntax (start..end)",
            ))
        }
    };

    let body = parse_block(&expr_for.body)?;

    Ok(Statement::For {
        var,
        start,
        end,
        body,
    })
}

fn parse_expr(expr: &SynExpr) -> syn::Result<Expr> {
    match expr {
        SynExpr::Lit(ExprLit { lit, .. }) => parse_lit(lit),

        SynExpr::Path(path) => {
            let name = path
                .path
                .get_ident()
                .ok_or_else(|| syn::Error::new(path.span(), "Expected identifier"))?
                .to_string();
            Ok(Expr::Var(name))
        }

        SynExpr::Binary(ExprBinary {
            op, left, right, ..
        }) => {
            let bin_op = parse_bin_op(op)?;
            Ok(Expr::Binary {
                op: bin_op,
                left: Box::new(parse_expr(left)?),
                right: Box::new(parse_expr(right)?),
            })
        }

        SynExpr::Unary(ExprUnary { op, expr, .. }) => {
            let unary_op = match op {
                syn::UnOp::Neg(_) => UnaryOp::Neg,
                syn::UnOp::Not(_) => UnaryOp::Not,
                _ => return Err(syn::Error::new(op.span(), "Unsupported unary operator")),
            };
            Ok(Expr::Unary {
                op: unary_op,
                operand: Box::new(parse_expr(expr)?),
            })
        }

        SynExpr::Call(ExprCall { func, args, .. }) => {
            let func_name = match &**func {
                SynExpr::Path(path) => path
                    .path
                    .get_ident()
                    .ok_or_else(|| syn::Error::new(path.span(), "Expected function name"))?
                    .to_string(),
                _ => {
                    return Err(syn::Error::new(
                        func.span(),
                        "Only simple function names supported",
                    ))
                }
            };

            let parsed_args: Vec<Expr> = args.iter().map(parse_expr).collect::<syn::Result<_>>()?;

            Ok(Expr::Call {
                func: func_name,
                args: parsed_args,
            })
        }

        SynExpr::Index(ExprIndex { expr, index, .. }) => {
            if let SynExpr::Path(path) = &**expr {
                let name = path
                    .path
                    .get_ident()
                    .ok_or_else(|| syn::Error::new(path.span(), "Expected identifier"))?
                    .to_string();
                let idx = parse_expr(index)?;
                match name.as_str() {
                    "STATE" => Ok(Expr::StateRead(Box::new(idx))),
                    "ATOMIC" => Ok(Expr::AtomicRead(Box::new(idx))),
                    _ => Err(syn::Error::new(
                        expr.span(),
                        "Only STATE[i] and ATOMIC[i] indexing supported",
                    )),
                }
            } else {
                Err(syn::Error::new(
                    expr.span(),
                    "Only STATE[i] and ATOMIC[i] indexing supported",
                ))
            }
        }

        SynExpr::Field(ExprField { base, member, .. }) => {
            let base_expr = parse_expr(base)?;
            let field_name = match member {
                syn::Member::Named(ident) => ident.to_string(),
                syn::Member::Unnamed(idx) => idx.index.to_string(),
            };
            Ok(Expr::Field {
                base: Box::new(base_expr),
                field: field_name,
            })
        }

        SynExpr::Cast(ExprCast { expr, ty, .. }) => {
            let inner = parse_expr(expr)?;
            let target_ty = parse_type(ty).ok_or_else(|| {
                syn::Error::new(ty.span(), "Unsupported cast type")
            })?;
            Ok(Expr::Cast {
                expr: Box::new(inner),
                ty: target_ty,
            })
        }

        SynExpr::Paren(ExprParen { expr, .. }) => parse_expr(expr),

        SynExpr::MethodCall(mc) => {
            // Convert method call to regular call
            // e.g., vec.xy() becomes xy(vec)
            let receiver = parse_expr(&mc.receiver)?;
            let method = mc.method.to_string();
            let mut args = vec![receiver];
            for arg in &mc.args {
                args.push(parse_expr(arg)?);
            }
            Ok(Expr::Call { func: method, args })
        }

        _ => Err(syn::Error::new(
            expr.span(),
            format!("Unsupported expression type: {:?}", expr),
        )),
    }
}

fn parse_lit(lit: &Lit) -> syn::Result<Expr> {
    match lit {
        Lit::Int(i) => {
            let value: i64 = i.base10_parse()?;
            Ok(Expr::LitInt(value))
        }
        Lit::Float(f) => {
            let value: f64 = f.base10_parse()?;
            Ok(Expr::LitFloat(value))
        }
        Lit::Bool(b) => Ok(Expr::LitBool(b.value)),
        _ => Err(syn::Error::new(lit.span(), "Unsupported literal type")),
    }
}

fn parse_bin_op(op: &syn::BinOp) -> syn::Result<BinOp> {
    match op {
        syn::BinOp::Add(_) => Ok(BinOp::Add),
        syn::BinOp::Sub(_) => Ok(BinOp::Sub),
        syn::BinOp::Mul(_) => Ok(BinOp::Mul),
        syn::BinOp::Div(_) => Ok(BinOp::Div),
        syn::BinOp::Rem(_) => Ok(BinOp::Rem),
        syn::BinOp::Eq(_) => Ok(BinOp::Eq),
        syn::BinOp::Ne(_) => Ok(BinOp::Ne),
        syn::BinOp::Lt(_) => Ok(BinOp::Lt),
        syn::BinOp::Le(_) => Ok(BinOp::Le),
        syn::BinOp::Gt(_) => Ok(BinOp::Gt),
        syn::BinOp::Ge(_) => Ok(BinOp::Ge),
        syn::BinOp::And(_) => Ok(BinOp::And),
        syn::BinOp::Or(_) => Ok(BinOp::Or),
        syn::BinOp::BitAnd(_) => Ok(BinOp::BitAnd),
        syn::BinOp::BitOr(_) => Ok(BinOp::BitOr),
        syn::BinOp::BitXor(_) => Ok(BinOp::BitXor),
        syn::BinOp::Shl(_) => Ok(BinOp::Shl),
        syn::BinOp::Shr(_) => Ok(BinOp::Shr),
        _ => Err(syn::Error::new(op.span(), "Unsupported binary operator")),
    }
}
