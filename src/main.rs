use std::{format, io::Write, unreachable};

fn main() -> Result<(), String> {
    test()
}

fn test() -> Result<(), String> {
    'loop_start: loop {
        let mut line = String::new();
        print!(">>> ");

        let _ = std::io::stdout().flush();
        let _ = std::io::stdin().read_line(&mut line).unwrap();

        let input_chars = line.trim_end().chars().collect::<Vec<_>>();

        if input_chars.len() == 0 {
            continue 'loop_start;
        }

        let mut tokenizer = Tokenizer::new(&input_chars);
        let tokens = tokenizer.tokenize();

        match tokens {
            Ok(token_list) => {
                let mut parser = Parser::new(&token_list);
                let ast = parser.parse();

                match ast {
                    Ok(result) => {
                        println!("{:}", result.calculate());
                    }
                    Err(err) => {
                        println!("{:?}", err);
                        continue 'loop_start;
                    }
                }
            }
            Err(err) => {
                println!("{:?}", err);
                continue 'loop_start;
            }
        }
    }
}

#[derive(PartialEq, Debug, Clone, Copy)]
enum KeywordToken {
    Let,
}
impl KeywordToken {
    fn get_all() -> [KeywordToken; 1] {
        [Self::Let]
    }
    fn value(&self) -> &str {
        match self {
            Self::Let => "let",
        }
    }
    fn is_equal<'a>(&self, check_with: &'a [char]) -> bool {
        check_with.iter().collect::<String>() == self.value()
    }
}

#[derive(PartialEq, Debug, Clone, Copy)]
enum SymbolToken {
    SmallBracketOpen,
    SmallBracketClose,
    Dot,
    Equal,
    UnderScore,
}
impl SymbolToken {
    fn get_all() -> [SymbolToken; 5] {
        [
            Self::SmallBracketOpen,
            Self::SmallBracketClose,
            Self::Dot,
            Self::Equal,
            Self::UnderScore,
        ]
    }
    fn value(&self) -> &str {
        match self {
            Self::SmallBracketOpen => "(",
            Self::SmallBracketClose => ")",
            Self::Dot => ".",
            Self::Equal => "=",
            Self::UnderScore => "_",
        }
    }
    fn is_equal<'a>(&self, check_with: &'a [char]) -> bool {
        check_with.iter().collect::<String>() == self.value()
    }
}
#[derive(PartialEq, Debug, Clone, Copy)]
enum OperatorToken {
    PLUS,
    MINUS,
    STAR,
    DIVIDE,
}
impl OperatorToken {
    fn get_all() -> [OperatorToken; 4] {
        [Self::PLUS, Self::MINUS, Self::STAR, Self::DIVIDE]
    }
    fn value(&self) -> &str {
        match self {
            Self::PLUS => "+",
            Self::MINUS => "-",
            Self::STAR => "*",
            Self::DIVIDE => "/",
        }
    }
    fn is_equal<'a>(&self, check_with: &'a [char]) -> bool {
        check_with.iter().collect::<String>() == self.value()
    }
}

#[derive(PartialEq, Debug, Clone, Copy)]
enum Token<'a> {
    Digit(&'a char, TokenInfo),
    Alphabet(&'a char, TokenInfo),
    Identifier(&'a [char], TokenInfo),

    Operator(OperatorToken, TokenInfo),
    Symbol(SymbolToken, TokenInfo),

    Keyword(KeywordToken, TokenInfo),

    WhiteSpace(TokenInfo),

    EOF(TokenInfo),

    Alphanumeric(&'a [char]),
}
impl<'a> Token<'a> {
    fn get_token_info(&self) -> &TokenInfo {
        match self {
            Self::Digit(_, token_info) => token_info,
            Self::Alphabet(_, token_info) => token_info,
            Self::Identifier(_, token_info) => token_info,
            Self::Operator(_, token_info) => token_info,
            Self::Symbol(_, token_info) => token_info,
            Self::Keyword(_, token_info) => token_info,
            Self::EOF(token_info) => token_info,

            Self::WhiteSpace(token_info) => token_info,
            Self::Alphanumeric(_) => unreachable!(),
        }
    }
}
#[derive(PartialEq, Debug, Clone, Copy)]
struct TokenInfo {
    line_number: usize,
    column_number: usize,
}
struct Tokenizer<'a> {
    input_string: &'a [char],
    index: usize,
    len: usize,
    column_number: usize,
    line_number: usize,
}

impl<'a> Tokenizer<'a> {
    fn new(input_string: &'a [char]) -> Self {
        Self {
            input_string: input_string,
            index: 0,
            len: input_string.len(),
            column_number: 1,
            line_number: 1,
        }
    }
    fn get_token_info(&self) -> TokenInfo {
        TokenInfo {
            line_number: self.line_number,
            column_number: self.column_number,
        }
    }

    fn advance(&mut self) {
        self.index += 1;
        self.column_number += 1;
    }

    fn peek(&self) -> Option<&char> {
        if self.index < self.len {
            Some(&self.input_string[self.index])
        } else {
            None
        }
    }

    fn reset_index_at(&mut self, new_index: usize) {
        let index_diff = self.index - new_index;
        self.index = new_index;
        self.column_number -= index_diff;
    }

    fn get_alphanumeric_token(&self, start_index: usize, end_index: usize) -> Option<Token<'a>> {
        if start_index >= end_index {
            None
        } else {
            Some(Token::Alphanumeric(
                &self.input_string[start_index..end_index],
            ))
        }
    }
    fn match_any_in_blob(&mut self, blob: &str) -> Option<Token<'a>> {
        let initial = self.index;
        match self.peek() {
            Some(&chr) if blob.contains(chr) => {
                self.advance();
                return self.get_alphanumeric_token(initial, self.index);
            }
            _ => {
                return None;
            }
        }
    }
    fn match_many_in_blob(&mut self, blob: &str) -> Option<Token<'a>> {
        let initial_index = self.index;
        loop {
            match self.peek() {
                Some(&chr) if blob.contains(chr) => self.advance(),
                _ => break,
            }
        }
        self.get_alphanumeric_token(initial_index, self.index)
    }
    fn match_exact_in_blob(&mut self, blob: &str) -> Option<Token<'a>> {
        let initial_index = self.index;
        for c in blob.chars() {
            match self.peek() {
                Some(&chr) if chr == c => self.advance(),
                _ => {
                    self.reset_index_at(initial_index);
                    return None;
                }
            }
        }
        self.get_alphanumeric_token(initial_index, self.index)
    }

    fn match_operator(&mut self) -> Option<Token<'a>> {
        let token_info = self.get_token_info();
        for operator_token in OperatorToken::get_all() {
            match self.match_exact_in_blob(operator_token.value()) {
                Some(Token::Alphanumeric(a)) if operator_token.is_equal(a) => {
                    return Some(Token::Operator(operator_token, token_info));
                }
                _ => {
                    continue;
                }
            }
        }
        None
    }
    fn match_symbol(&mut self) -> Option<Token<'a>> {
        let token_info = self.get_token_info();
        for symbol_token in SymbolToken::get_all() {
            match self.match_exact_in_blob(symbol_token.value()) {
                Some(Token::Alphanumeric(a)) if symbol_token.is_equal(a) => {
                    return Some(Token::Symbol(symbol_token, token_info));
                }
                _ => {
                    continue;
                }
            }
        }
        None
    }
    fn match_whitespace(&mut self) -> Option<Token<'a>> {
        let token_info = self.get_token_info();
        match self.match_many_in_blob(" ") {
            Some(_) => Some(Token::WhiteSpace(token_info)),
            _ => None,
        }
    }

    fn match_digit(&mut self) -> Option<Token<'a>> {
        let token_info = self.get_token_info();
        match self.match_any_in_blob("0123456789") {
            Some(Token::Alphanumeric(chr)) => Some(Token::Digit(chr.first().unwrap(), token_info)),
            _ => None,
        }
    }
    fn match_alphabet(&mut self) -> Option<Token<'a>> {
        let blob = ('a'..='z').chain('A'..='Z').collect::<String>();
        let token_info = self.get_token_info();
        match self.match_any_in_blob(&blob) {
            Some(Token::Alphanumeric(chr)) => {
                Some(Token::Alphabet(chr.first().unwrap(), token_info))
            }
            _ => None,
        }
    }
    fn match_identifier(&mut self) -> Option<Token<'a>> {
        let blob = ('a'..='z').chain('A'..='Z').collect::<String>();
        let token_info = self.get_token_info();
        match self.match_many_in_blob(&blob) {
            Some(Token::Alphanumeric(chr)) => Some(Token::Identifier(chr, token_info)),
            _ => None,
        }
    }
    fn match_keyword(&mut self) -> Option<Token<'a>> {
        let token_info = self.get_token_info();
        let initial_index = self.index;
        for keyword_token in KeywordToken::get_all() {
            match self.match_exact_in_blob(keyword_token.value()) {
                Some(Token::Alphanumeric(a)) if keyword_token.is_equal(a) => {
                    if self.peek() == Some(&' ') {
                        return Some(Token::Keyword(keyword_token, token_info));
                    } else {
                        self.reset_index_at(initial_index);
                        return None;
                    }
                }
                _ => {
                    continue;
                }
            }
        }
        None
    }

    fn tokenize(&mut self) -> Result<Vec<Token>, Vec<String>> {
        let mut tokens = Vec::new();
        let mut tokenization_errors = vec![];
        while self.peek().is_some() {
            if let Some(token) = self.match_keyword() {
                tokens.push(token);
                continue;
            }
            if let Some(token) = self.match_digit() {
                tokens.push(token);
                continue;
            }
            if let Some(token) = self.match_identifier() {
                tokens.push(token);
                continue;
            }
            if let Some(token) = self.match_operator() {
                tokens.push(token);
                continue;
            }
            if let Some(token) = self.match_symbol() {
                tokens.push(token);
                continue;
            }
            if let Some(token) = self.match_whitespace() {
                tokens.push(token);
                continue;
            }
            let token_info = self.get_token_info();
            tokenization_errors.push(format!(
                "Tokenizing Error: Unknown token at line {}, column {}",
                token_info.line_number, token_info.column_number
            ));
            self.advance();
        }
        if tokenization_errors.len() > 0 {
            Err(tokenization_errors)
        } else {
            let token_info = self.get_token_info();
            tokens.push(Token::EOF(token_info));
            println!("{:?}", tokens);
            Result::Ok(tokens)
        }
    }
}

struct Parser<'a> {
    tokens: &'a [Token<'a>],
    index: usize,
    len: usize,
}

impl<'a> Parser<'a> {
    fn new(tokens: &'a [Token<'a>]) -> Self {
        Parser {
            tokens: tokens,
            index: 0,
            len: tokens.len(),
        }
    }

    fn parse(&mut self) -> Result<Either, String> {
        let return_value = self.parse_let_statement()?;
        match self.peek() {
            Some(Token::EOF(_)) => Ok(return_value),
            Some(token) => Err(format!(
                "Parsing Error at: {} {}",
                token.get_token_info().line_number,
                token.get_token_info().column_number
            )),
            None => Err(format!("Parsing Error: None value encountered")),
        }
    }

    fn advance(&mut self) -> Option<&Token> {
        if self.index < self.len {
            let return_value = &self.tokens[self.index];
            self.index += 1;
            Some(return_value)
        } else {
            None
        }
    }

    fn peek(&self) -> Option<&Token<'a>> {
        if self.index < self.len {
            Some(&self.tokens[self.index])
        } else {
            None
        }
    }

    fn consume_optional_whitespace(&mut self) {
        if let Some(Token::WhiteSpace(_)) = self.peek() {
            self.advance();
        }
    }

    fn parse_let_statement(&mut self) -> Result<Either, String> {
        match self.peek().cloned() {
            Some(Token::Keyword(keyword, token_info))
                if keyword.value() == KeywordToken::Let.value() =>
            {
                self.advance();
                self.advance(); //consume whitespace
                match self.peek().cloned() {
                    Some(Token::Identifier(identifier, iden_token_info)) => {
                        self.advance();
                        self.consume_optional_whitespace();
                        match self.peek().cloned() {
                            Some(Token::Symbol(symbol, _))
                                if symbol.value() == SymbolToken::Equal.value() =>
                            {
                                self.advance();
                                self.consume_optional_whitespace();
                                // self.parse_term()
                                Ok(LetStatement::new(
                                    identifier.iter().collect(),
                                    self.parse_term()?,
                                ))
                            }
                            Some(token) => Err(format!(
                                "Parsing Error: No equal found at line {}, column {}",
                                token.get_token_info().line_number,
                                token.get_token_info().column_number
                            )),
                            None => Err(format!(
                                "Parsing Error: None value encountered at line {}, {}",
                                iden_token_info.line_number, iden_token_info.column_number
                            )),
                        }
                    }
                    Some(token) => Err(format!(
                        "Parsing Error: No closing bracket found at line {}, column {}",
                        token.get_token_info().line_number,
                        token.get_token_info().column_number
                    )),
                    None => Err(format!(
                        "Parsing Error: No closing bracket found at line {}, column {}",
                        token_info.line_number, token_info.column_number
                    )),
                }
            }
            _ => self.parse_term(),
        }
    }

    fn parse_term(&mut self) -> Result<Either, String> {
        let mut first_expression = self.parse_prod()?;

        loop {
            match self.peek().cloned() {
                Some(Token::Operator(
                    operator_token @ (OperatorToken::PLUS | OperatorToken::MINUS),
                    _,
                )) => {
                    self.advance();
                    let expr =
                        BinaryExpression::new(first_expression, self.parse_prod()?, operator_token);
                    first_expression = expr;
                    continue;
                }
                Some(Token::WhiteSpace(_)) => {
                    self.advance();
                    continue;
                }
                _ => {
                    return Ok(first_expression);
                }
            }
        }
    }

    fn parse_prod(&mut self) -> Result<Either, String> {
        let mut first_expression = self.parse_unary()?;

        loop {
            match self.peek().cloned() {
                Some(Token::Operator(
                    operator_token @ (OperatorToken::STAR | OperatorToken::DIVIDE),
                    _,
                )) => {
                    self.advance();
                    let expr = BinaryExpression::new(
                        first_expression,
                        self.parse_unary()?,
                        operator_token,
                    );
                    first_expression = expr;
                    continue;
                }
                Some(Token::WhiteSpace(_)) => {
                    self.advance();
                    continue;
                }
                _ => {
                    return Ok(first_expression);
                }
            }
        }
    }

    fn parse_unary(&mut self) -> Result<Either, String> {
        match self.peek().cloned() {
            Some(Token::Operator(operator @ (OperatorToken::PLUS | OperatorToken::MINUS), _)) => {
                self.advance();
                let expr = UnaryExpression::new(self.parse_unary()?, operator);
                Ok(expr)
            }
            Some(Token::WhiteSpace(_)) => {
                self.advance();
                self.parse_unary()
            }
            _ => self.parse_bracket(),
        }
    }

    fn parse_bracket(&mut self) -> Result<Either, String> {
        match self.peek().cloned() {
            Some(Token::Symbol(SymbolToken::SmallBracketOpen, _)) => {
                self.advance();
                let val = self.parse_term()?;
                match self.peek().cloned() {
                    Some(Token::Symbol(SymbolToken::SmallBracketClose, _)) => {
                        self.advance();
                        Ok(val)
                    }
                    Some(Token::WhiteSpace(_)) => {
                        self.advance();
                        self.parse_bracket()
                    }
                    Some(token) => Err(format!(
                        "Parsing Error: No closing bracket found at line {}, column {}",
                        token.get_token_info().line_number,
                        token.get_token_info().column_number
                    )),
                    None => Err(format!("Parsing Error: None value encountered")),
                }
            }
            _ => self.parse_number(),
        }
    }

    fn parse_number(&mut self) -> Result<Either, String> {
        match self.peek().cloned() {
            Some(Token::Digit(first_digit, _)) => {
                let mut digits = vec![first_digit];
                let mut num_of_dot = 0;
                self.advance();
                loop {
                    match self.peek().cloned() {
                        Some(Token::Symbol(SymbolToken::Dot, _)) if num_of_dot < 1 => {
                            self.advance();
                            num_of_dot += 1;
                            digits.push(&'.');
                        }
                        Some(Token::Symbol(SymbolToken::Dot, token_info_symbol))
                            if num_of_dot >= 1 =>
                        {
                            return Err(format!(
                "Parsing Error: Decimal(.) parsing error encountered at line {}, column {}",
                token_info_symbol.line_number,
                token_info_symbol.column_number ,
            ))
                        }
                        Some(Token::Digit(val, _)) => {
                            self.advance();
                            digits.push(val);
                        }

                        _ => {
                            break;
                        }
                    }
                }
                Ok(Either::Number(
                    digits
                        .into_iter()
                        .collect::<String>()
                        .parse::<f64>()
                        .unwrap(),
                ))
            }
            Some(token) => Err(format!(
                "Parsing Error: Value other than number encountered at line {}, column {}",
                token.get_token_info().line_number,
                token.get_token_info().column_number,
            )),
            None => Err(format!("Parsing Error: None value encountered")),
        }
    }
}

#[derive(Debug)]
enum Either {
    Number(f64),
    BinaryExpression(Box<BinaryExpression>),
    UnaryExpression(Box<UnaryExpression>),
    LetStatement(Box<LetStatement>),
}

impl Either {
    fn calculate(&self) -> f64 {
        match self {
            Self::Number(val) => *val,

            Self::BinaryExpression(boxed_expression) => match &**boxed_expression {
                BinaryExpression {
                    left,
                    right,
                    operator,
                } => match operator {
                    OperatorToken::PLUS => left.calculate() + right.calculate(),
                    OperatorToken::MINUS => left.calculate() - right.calculate(),
                    OperatorToken::STAR => left.calculate() * right.calculate(),
                    OperatorToken::DIVIDE => left.calculate() / right.calculate(),
                },
            },
            Self::UnaryExpression(boxed_expression) => match &**boxed_expression {
                UnaryExpression { val, operator } => match operator {
                    OperatorToken::PLUS => 1.0 * val.calculate(),
                    OperatorToken::MINUS | _ => -1.0 * val.calculate(),
                },
            },
            Self::LetStatement(boxed_let_statement) => match &**boxed_let_statement {
                LetStatement { lvalue, rvalue } => {
                    println!("{:?}", boxed_let_statement);
                    rvalue.calculate()
                }
            },
        }
    }
}

#[derive(Debug)]
struct BinaryExpression {
    left: Either,
    right: Either,
    operator: OperatorToken,
}
impl BinaryExpression {
    fn new(left: Either, right: Either, operator: OperatorToken) -> Either {
        Either::BinaryExpression(Box::new(Self {
            left: left,
            right: right,
            operator: operator,
        }))
    }
}
#[derive(Debug)]
struct UnaryExpression {
    val: Either,
    operator: OperatorToken,
}
impl UnaryExpression {
    fn new(val: Either, operator: OperatorToken) -> Either {
        Either::UnaryExpression(Box::new(Self {
            val: val,
            operator: operator,
        }))
    }
}

#[derive(Debug)]
struct LetStatement {
    lvalue: String,
    rvalue: Either,
}

impl LetStatement {
    fn new(lvalue: String, rvalue: Either) -> Either {
        Either::LetStatement(Box::new(Self {
            lvalue: lvalue,
            rvalue: rvalue,
        }))
    }
}
