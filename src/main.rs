use std::{collections::HashMap, format, io::Write, println, rc::Rc, unreachable, vec};

fn main() {
    let prompt = "
    let a = 1
    let b = 1
    let counter = 1
    while {counter < 10} {
                            let c = a + b
                            a = b
                            b = c
                            counter = counter + 1
                         }
    counter a b
    ";
    println!("Fibonacci Calculator:\n{}", prompt);

    let mut interpreter = Interpreter::new();

    interpreter.interpret(prompt);
    interpreter.interactive_prompt();
}

#[derive(PartialEq, Debug, Clone, Copy)]
enum Keywords {
    Let,
    True,
    False,
    While,
    If,
    Else,
}
impl Keywords {
    fn get_all() -> [Keywords; 6] {
        [
            Self::Let,
            Self::True,
            Self::False,
            Self::While,
            Self::If,
            Self::Else,
        ]
    }
    fn lexeme(&self) -> &str {
        match self {
            Self::Let => "let",
            Self::True => "true",
            Self::False => "false",
            Self::While => "while",
            Self::If => "if",
            Self::Else => "else",
        }
    }
}

#[derive(PartialEq, Debug, Clone, Copy)]
enum Symbols {
    SmallBracketOpen,
    SmallBracketClose,
    Dot,
    Equal,
    // UnderScore,
    SemiColon,
    CurlyBracketOpen,
    CurlyBracketClose,
}
impl Symbols {
    fn get_all() -> [Symbols; 7] {
        [
            Self::SmallBracketOpen,
            Self::SmallBracketClose,
            Self::Dot,
            Self::Equal,
            // Self::UnderScore,
            Self::SemiColon,
            Self::CurlyBracketOpen,
            Self::CurlyBracketClose,
        ]
    }
    fn lexeme(&self) -> &str {
        match self {
            Self::SmallBracketOpen => "(",
            Self::SmallBracketClose => ")",
            Self::Dot => ".",
            Self::Equal => "=",
            // Self::UnderScore => "_",
            Self::SemiColon => ";",
            Self::CurlyBracketOpen => "{",
            Self::CurlyBracketClose => "}",
        }
    }
}
#[allow(non_camel_case_types)]
#[derive(PartialEq, Debug, Clone, Copy)]
enum Operators {
    PLUS,
    MINUS,
    STAR,
    DIVIDE,

    LOGICAL_AND,
    LOGICAL_OR,

    LOGICAL_NOT,

    LESS_THAN,
    GREATER_THAN,
    EQUAL_TO,
}
impl Operators {
    fn get_all() -> [Operators; 10] {
        [
            Self::PLUS,
            Self::MINUS,
            Self::STAR,
            Self::DIVIDE,
            Self::LOGICAL_AND,
            Self::LOGICAL_OR,
            Self::LOGICAL_NOT,
            Self::LESS_THAN,
            Self::GREATER_THAN,
            Self::EQUAL_TO,
        ]
    }
    fn lexeme(&self) -> &str {
        match self {
            Self::PLUS => "+",
            Self::MINUS => "-",
            Self::STAR => "*",
            Self::DIVIDE => "/",
            Self::LOGICAL_AND => "&&",
            Self::LOGICAL_OR => "||",
            Self::LOGICAL_NOT => "!",
            Self::LESS_THAN => "<",
            Self::GREATER_THAN => ">",
            Self::EQUAL_TO => "==",
        }
    }
    // output type after application of operator
    fn tipe(&self) -> &LanguageType {
        match self {
            Self::PLUS | Self::MINUS | Self::STAR | Self::DIVIDE => &LanguageType::Number,
            Self::LOGICAL_AND
            | Self::LOGICAL_OR
            | Self::LOGICAL_NOT
            | Self::LESS_THAN
            | Self::GREATER_THAN
            | Self::EQUAL_TO => &LanguageType::Boolean,
        }
    }
}

#[derive(PartialEq, Debug, Clone)]
enum Token {
    DigitToken {
        lexeme: Rc<str>,
        token_info: Rc<TokenInfo>,
    },
    AlphabetToken {
        lexeme: Rc<str>,
        token_info: Rc<TokenInfo>,
    },
    IdentifierToken {
        lexeme: Rc<str>,
        token_info: Rc<TokenInfo>,
    },
    OperatorToken {
        operator: Operators,
        token_info: Rc<TokenInfo>,
    },
    SymbolToken {
        symbol: Symbols,
        token_info: Rc<TokenInfo>,
    },
    KeywordToken {
        keyword: Keywords,
        token_info: Rc<TokenInfo>,
    },

    WhiteSpaceToken {
        token_info: Rc<TokenInfo>,
    },
    NewLineToken {
        token_info: Rc<TokenInfo>,
    },
    EOFToken {
        token_info: Rc<TokenInfo>,
    },
}
impl Token {
    fn get_token_info(&self) -> Rc<TokenInfo> {
        match self {
            Self::DigitToken {
                lexeme: _,
                token_info,
            } => token_info.clone(),
            Self::AlphabetToken {
                lexeme: _,
                token_info,
            } => token_info.clone(),
            Self::IdentifierToken {
                lexeme: _,
                token_info,
            } => token_info.clone(),

            Self::OperatorToken {
                operator: _,
                token_info,
            } => token_info.clone(),
            Self::SymbolToken {
                symbol: _,
                token_info,
            } => token_info.clone(),
            Self::KeywordToken {
                keyword: _,
                token_info,
            } => token_info.clone(),

            Self::WhiteSpaceToken { token_info } => token_info.clone(),
            Self::NewLineToken { token_info } => token_info.clone(),

            Self::EOFToken { token_info } => token_info.clone(),
        }
    }
    fn value(&self) -> Rc<str> {
        match self {
            Self::DigitToken {
                lexeme,
                token_info: _,
            } => lexeme.clone(),
            Self::AlphabetToken {
                lexeme,
                token_info: _,
            } => lexeme.clone(),
            Self::IdentifierToken {
                lexeme,
                token_info: _,
            } => lexeme.clone(),

            Self::OperatorToken {
                operator,
                token_info: _,
            } => operator.lexeme().into(),

            Self::SymbolToken {
                symbol,
                token_info: _,
            } => symbol.lexeme().into(),
            Self::KeywordToken {
                keyword,
                token_info: _,
            } => keyword.lexeme().into(),

            Self::WhiteSpaceToken { token_info: _ } => " ".into(),
            Self::NewLineToken { token_info: _ } => "\n".into(),

            Self::EOFToken { token_info: _ } => "<EOF>".into(),
        }
    }
}
#[derive(PartialEq, Debug, Clone, Copy)]
struct TokenInfo {
    line_number: usize,
    column_number: usize,
}
struct Tokenizer {
    input_string: Rc<str>,
    index: usize,
    column_number: usize,
    line_number: usize,
}

impl Tokenizer {
    fn new(input_string: &str) -> Self {
        Self {
            input_string: input_string.into(),
            index: 0,
            column_number: 1,
            line_number: 1,
        }
    }
    fn get_token_info(&self) -> Rc<TokenInfo> {
        Rc::new(TokenInfo {
            line_number: self.line_number,
            column_number: self.column_number,
        })
    }

    fn advance(&mut self) {
        self.index += 1;
        self.column_number += 1;
    }
    fn break_lines(&mut self, lines: usize) {
        self.line_number += lines;
        self.reset_column();
    }

    fn peek(&self) -> Option<char> {
        self.input_string.chars().nth(self.index)
    }

    fn reset_index_at(&mut self, new_index: usize) {
        let index_diff = self.index - new_index;
        self.index = new_index;
        self.column_number -= index_diff;
    }
    fn reset_column(&mut self) {
        self.column_number = 1;
    }

    fn match_any_in_blob(&mut self, blob: &str) -> Option<Rc<str>> {
        let initial = self.index;
        match self.peek() {
            Some(chr) if blob.contains(chr) => {
                self.advance();
                return Some(self.input_string[initial..self.index].into());
            }
            _ => {
                return None;
            }
        }
    }
    fn match_many_in_blob(&mut self, blob: &str) -> Option<Rc<str>> {
        let initial_index = self.index;
        loop {
            match self.peek() {
                Some(chr) if blob.contains(chr) => self.advance(),
                _ => break,
            }
        }
        if initial_index != self.index {
            Some(self.input_string[initial_index..self.index].into())
        } else {
            None
        }
    }
    fn match_exact_in_blob(&mut self, blob: &str) -> Option<Rc<str>> {
        let initial_index = self.index;
        for c in blob.chars() {
            match self.peek() {
                Some(chr) if chr == c => self.advance(),
                _ => {
                    self.reset_index_at(initial_index);
                    return None;
                }
            }
        }
        Some(self.input_string[initial_index..self.index].into())
    }

    fn match_operator(&mut self) -> Option<Token> {
        let token_info = self.get_token_info();
        for operator in Operators::get_all() {
            match self.match_exact_in_blob(operator.lexeme()) {
                Some(_) => {
                    return Some(Token::OperatorToken {
                        operator,
                        token_info,
                    });
                }
                _ => {
                    continue;
                }
            }
        }
        None
    }
    fn match_symbol(&mut self) -> Option<Token> {
        let token_info = self.get_token_info();
        for symbol in Symbols::get_all() {
            match self.match_exact_in_blob(symbol.lexeme()) {
                Some(_) => {
                    return Some(Token::SymbolToken { symbol, token_info });
                }
                _ => {
                    continue;
                }
            }
        }
        None
    }
    fn match_whitespace(&mut self) -> Option<Token> {
        let token_info = self.get_token_info();
        match self.match_many_in_blob(" ") {
            Some(_) => Some(Token::WhiteSpaceToken { token_info }),
            _ => None,
        }
    }
    fn match_new_lines(&mut self) -> Option<Token> {
        let token_info = self.get_token_info();
        match self.match_many_in_blob("\n") {
            Some(match_string) => {
                let no_of_new_lines = match_string.len();
                self.break_lines(no_of_new_lines);

                Some(Token::NewLineToken { token_info })
            }
            _ => None,
        }
    }

    fn match_digit(&mut self) -> Option<Token> {
        let token_info = self.get_token_info();
        match self.match_any_in_blob("0123456789") {
            Some(lexeme) => Some(Token::DigitToken { lexeme, token_info }),
            _ => None,
        }
    }
    fn match_alphabet(&mut self) -> Option<Token> {
        let blob = ('a'..='z').chain('A'..='Z').collect::<String>();
        let token_info = self.get_token_info();
        match self.match_any_in_blob(&blob) {
            Some(lexeme) => Some(Token::AlphabetToken { lexeme, token_info }),
            _ => None,
        }
    }
    fn match_identifier_or_keyword(&mut self) -> Option<Token> {
        let blob = ('a'..='z').chain('A'..='Z').collect::<String>();
        let token_info = self.get_token_info();
        match self.match_many_in_blob(&blob) {
            Some(lexeme) => {
                for keyword in Keywords::get_all().into_iter() {
                    if lexeme == keyword.lexeme().into() {
                        return Some(Token::KeywordToken {
                            keyword,
                            token_info,
                        });
                    }
                }
                Some(Token::IdentifierToken { lexeme, token_info })
            }
            _ => None,
        }
    }

    fn tokenize(&mut self) -> Result<Vec<Token>, Vec<String>> {
        let mut tokens = Vec::new();
        let mut tokenization_errors = vec![];
        while self.peek().is_some() {
            if let Some(token) = self.match_identifier_or_keyword() {
                tokens.push(token);
                continue;
            }
            if let Some(token) = self.match_digit() {
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
            if let Some(token) = self.match_new_lines() {
                // tokens.push(token);
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
            tokens.push(Token::EOFToken { token_info });
            // println!("{:?}", tokens);
            Result::Ok(tokens)
        }
    }
}

struct Parser<'a> {
    tokens: &'a [Token],
    index: usize,
    back_track_index: usize,
    len: usize,
    block_position: usize,
    num_identifiers_in_block: Vec<usize>,
    identifiers_list: Option<Vec<AST>>,
}

impl<'a> Parser<'a> {
    fn new(tokens: &'a [Token]) -> Self {
        Parser {
            tokens: tokens,
            index: 0,
            back_track_index: 0,
            len: tokens.len(),
            block_position: 0,
            num_identifiers_in_block: vec![0],
            identifiers_list: None,
        }
    }
    fn increment_num_identifier_in_current_block(&mut self) {
        let block_postion = self.get_block_position();
        self.num_identifiers_in_block[block_postion] += 1;
    }

    fn get_block_position(&self) -> usize {
        self.block_position
    }
    fn increment_block_position(&mut self) {
        self.block_position += 1;
        self.num_identifiers_in_block.push(0)
    }
    fn decrement_block_position(&mut self) {
        let current_block_position = self.block_position;
        let num_identifiers_to_pop = self.num_identifiers_in_block[current_block_position];
        for _ in 0..num_identifiers_to_pop {
            self.identifiers_list.as_mut().unwrap().pop();
        }
        self.block_position -= 1;
        self.num_identifiers_in_block.pop();
    }

    fn remember_index(&mut self) {
        self.back_track_index = self.index;
    }

    fn back_track(&mut self) {
        self.index = self.back_track_index;
    }

    fn get_identifiers_list(self) -> Option<Vec<AST>> {
        self.identifiers_list
    }

    fn push_identifier(&mut self, identifier: AST) {
        if self.identifiers_list.is_none() {
            self.identifiers_list = Some(vec![identifier]);
        } else {
            self.identifiers_list.as_mut().unwrap().push(identifier);
        }

        self.increment_num_identifier_in_current_block();
    }

    fn set_previous_identifiers_list(&mut self, previous_identifier_list: Option<Vec<AST>>) {
        self.identifiers_list = previous_identifier_list
    }

    fn get_reference_to_identifier(&self, name: &Rc<str>) -> Option<AST> {
        match &self.identifiers_list {
            Some(identifiers) => {
                for identifier in identifiers.iter().rev() {
                    if let Some(name_to_match) = identifier.get_identifier_name() {
                        if name_to_match == *name {
                            return identifier.clone_identifier();
                        }
                    }
                }
            }
            None => return None,
        }
        None
    }

    fn modify_identifier_type(&mut self, identifier_to_modify: &AST, tipe_value: LanguageType) {
        match self.identifiers_list.as_deref_mut() {
            Some(identifiers) => {
                for identifier in identifiers.iter_mut().rev() {
                    if identifier.get_identifier_name()
                        == identifier_to_modify.get_identifier_name()
                    {
                        match identifier.clone_and_modify_identifier(tipe_value) {
                            Some(cloned_identifier) => {
                                *identifier = cloned_identifier;
                            }
                            None => continue,
                        }
                    }
                }
            }
            _ => {}
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

    fn peek(&self) -> Option<Token> {
        if self.index < self.len {
            Some(&self.tokens[self.index]).cloned()
        } else {
            None
        }
    }

    fn consume_optional_semicolon(&mut self) -> bool {
        match self.peek() {
            Some(Token::SymbolToken {
                symbol: Symbols::SemiColon,
                token_info: _,
            }) => {
                self.advance();
                true
            }
            None | Some(_) => false,
        }
    }

    fn consume_optional_whitespace(&mut self) -> bool {
        match self.peek() {
            Some(Token::WhiteSpaceToken { token_info: _ }) => {
                self.advance();
                true
            }
            None | Some(_) => false,
        }
    }

    fn parse(&mut self) -> Result<Vec<AST>, ParseError> {
        let mut statements = Vec::new();
        loop {
            let parsed_statement = self.parse_next_statement()?;
            self.consume_optional_semicolon();
            self.consume_optional_whitespace();
            statements.push(parsed_statement);

            if self.peek().is_none() {
                break;
            }
        }
        // println!("{:?}", statements);

        Ok(statements)
    }

    fn parse_next_statement(&mut self) -> Result<AST, ParseError> {
        self.parse_halt_statement()
            .or_else(|e| match e {
                ParseError::StructureUnimplementedError(_) => self.parse_while_statement(),
                ParseError::TypeError(_)
                | ParseError::IdentifierError(_)
                | ParseError::SyntaxError(_) => Err(e),
            })
            .or_else(|e| match e {
                ParseError::StructureUnimplementedError(_) => self.parse_let_statement(),
                ParseError::TypeError(_)
                | ParseError::IdentifierError(_)
                | ParseError::SyntaxError(_) => Err(e),
            })
            .or_else(|e| match e {
                ParseError::StructureUnimplementedError(_) => self.parse_assignment_statement(),
                ParseError::TypeError(_)
                | ParseError::IdentifierError(_)
                | ParseError::SyntaxError(_) => Err(e),
            })
            .or_else(|e| match e {
                ParseError::StructureUnimplementedError(_) => self.parse_expression(),
                ParseError::TypeError(_)
                | ParseError::IdentifierError(_)
                | ParseError::SyntaxError(_) => Err(e),
            })
            .or_else(|e| match e {
                e @ (ParseError::StructureUnimplementedError(_)
                | ParseError::TypeError(_)
                | ParseError::IdentifierError(_)
                | ParseError::SyntaxError(_)) => Err(e),
            })
    }
    fn parse_halt_statement(&mut self) -> Result<AST, ParseError> {
        match self.peek() {
            Some(Token::EOFToken { token_info: _ }) => {
                self.advance();
                Ok(AST::HaltStatement)
            }
            Some(token) => Err(ParseError::StructureUnimplementedError(format!(
                "Parsing Error: Cannot parse halt statement found at line {}, column {}",
                token.get_token_info().line_number,
                token.get_token_info().column_number,
            ))),
            None => unreachable!(),
        }
    }
    fn parse_while_statement(&mut self) -> Result<AST, ParseError> {
        self.consume_optional_whitespace();
        match self.peek() {
            Some(Token::KeywordToken {
                keyword: Keywords::While,
                token_info,
            }) => {
                self.advance();
                self.consume_optional_whitespace();
                let condition_expression = self.parse_logical_term()?;
                if condition_expression.tipe() == LanguageType::Boolean {
                    self.consume_optional_whitespace();
                    let block_expression = self.parse_logical_term()?;
                    Ok(AST::new_while_statement(
                        condition_expression,
                        block_expression,
                        &token_info,
                    ))
                } else {
                    let token_info = condition_expression.token_info();
                    Err(ParseError::TypeError(format!(
                        "Type Error: Type mismatched found at line {}, column {}\n Condition expression must be boolean",
                        token_info.line_number, token_info.column_number
                    )))
                }
            }
            Some(token) => Err(ParseError::StructureUnimplementedError(format!(
                "Parsing Error: Cannot parse while statement found at line {}, column {}",
                token.get_token_info().line_number,
                token.get_token_info().column_number,
            ))),
            _ => unreachable!(),
        }
    }

    fn parse_let_statement(&mut self) -> Result<AST, ParseError> {
        self.consume_optional_whitespace();
        match self.peek() {
            Some(Token::KeywordToken {
                keyword: Keywords::Let,
                token_info,
            }) => {
                self.advance();
                self.advance(); //consume whitespace
                match self.peek() {
                    Some(Token::IdentifierToken {
                        lexeme,
                        token_info: identifier_token_info,
                    }) => {
                        self.advance();
                        self.consume_optional_whitespace();
                        match self.peek() {
                            Some(Token::SymbolToken {
                                symbol: Symbols::Equal,
                                token_info: _,
                            }) => {
                                self.advance();
                                self.consume_optional_whitespace();
                                let term = self.parse_expression()?;
                                let tipe = term.tipe();

                                match tipe {
                                    LanguageType::Void => Err(ParseError::TypeError(format!(
                                        "Type Error: Void type found at line {}, column {}",
                                        token_info.line_number, token_info.column_number
                                    ))),
                                    _ => {
                                        self.push_identifier(AST::new_identifier(
                                            &lexeme,
                                            self.get_block_position(),
                                            tipe,
                                            &identifier_token_info,
                                        ));
                                        Ok(AST::new_let_statement(&lexeme, term, &token_info))
                                    }
                                }
                            }
                            Some(_) => {
                                self.push_identifier(AST::new_identifier(
                                    &lexeme,
                                    self.get_block_position(),
                                    LanguageType::Untyped,
                                    &identifier_token_info,
                                ));
                                Ok(AST::new_unassigned_let_statement(&lexeme, &token_info))
                            }
                            None => Err(ParseError::SyntaxError(format!(
                                "Parsing Error: None value encountered at line {}, {}",
                                token_info.line_number, token_info.column_number
                            ))),
                        }
                    }
                    Some(token) => Err(ParseError::SyntaxError(format!(
                        "Syntax Error: No identifier found at line {}, column {}",
                        token.get_token_info().line_number,
                        token.get_token_info().column_number
                    ))),
                    None => Err(ParseError::SyntaxError(format!(
                        "Syntax Error: None value found at line {}, column {}",
                        token_info.line_number, token_info.column_number
                    ))),
                }
            }
            Some(token) => Err(ParseError::StructureUnimplementedError(format!(
                "Parsing Error: Cannot parse let statement at line {}, column {}",
                token.get_token_info().line_number,
                token.get_token_info().column_number
            ))),
            _ => unreachable!(),
        }
    }
    fn parse_assignment_statement(&mut self) -> Result<AST, ParseError> {
        self.consume_optional_whitespace();
        self.remember_index();
        match self.peek() {
            Some(Token::IdentifierToken { lexeme, token_info }) => {
                match self.get_reference_to_identifier(&lexeme) {
                    Some(identifier) => {
                        self.advance();
                        self.consume_optional_whitespace();
                        match self.peek() {
                            Some(Token::SymbolToken{symbol: Symbols::Equal, token_info:_}) =>
                            {
                                self.advance();
                                self.consume_optional_whitespace();
                                let term = self.parse_expression()?;
                                if term.tipe() == identifier.tipe() || identifier.tipe() == LanguageType::Untyped{
                                    self.modify_identifier_type(&identifier, term.tipe());
                                    Ok(AST::new_assignment_statement(&lexeme, term, &token_info))
                                } else {
                                    let token_info = term.token_info();
                                    Err(ParseError::TypeError(format!(
                                        "Type Error: Type mismatched, found at line {}, column {}",
                                        token_info.line_number, token_info.column_number
                                    )))
                                }
                            }
                            Some(other_token) => {
                                self.back_track();
                                Err(ParseError::StructureUnimplementedError(format!(
                                "Parsing Error: Cannot parse assignment statement found at line {}, column {}",
                                other_token.get_token_info().line_number,
                                other_token.get_token_info().column_number
                            )))
                            }
                            None => Err(ParseError::StructureUnimplementedError(format!(
                                "Parsing Error: Cannot parse assignment statement found at line {}, column {}",
                                token_info.line_number,
                                token_info.column_number
                            ))),
                        }
                    }
                    None => Err(ParseError::IdentifierError(format!(
                        "Identifier Error: use of undeclared variable '{}' encountered at line {}, column {}",
                        lexeme,
                        token_info.line_number,
                        token_info.column_number
                    ))),
                }
            }
            Some(other_token) => Err(ParseError::StructureUnimplementedError(format!(
                "Parsing Error: Cannot parse assignment statement at line {}, column {}",
                other_token.get_token_info().line_number,
                other_token.get_token_info().column_number
            ))),
            _ => unreachable!(),
        }
    }

    fn parse_expression(&mut self) -> Result<AST, ParseError> {
        self.parse_if_else_expression().or_else(|e| match e {
            ParseError::StructureUnimplementedError(_) => self.parse_logical_term(),
            ParseError::TypeError(_)
            | ParseError::IdentifierError(_)
            | ParseError::SyntaxError(_) => Err(e),
        })
    }

    fn parse_if_else_expression(&mut self) -> Result<AST, ParseError> {
        self.consume_optional_whitespace();
        match self.peek() {
            Some(Token::KeywordToken {
                keyword: Keywords::If,
                token_info,
            }) => {
                self.advance();
                self.consume_optional_whitespace();
                let cond = self.parse_logical_term()?;
                if cond.tipe() == LanguageType::Boolean {
                    self.consume_optional_whitespace();
                    let if_block = self.parse_logical_term()?;
                    let if_block_tipe = if_block.tipe();

                    self.consume_optional_whitespace();

                    match self.peek(){
                        Some(Token::KeywordToken{keyword: Keywords::Else, token_info:_}) => {
                            self.advance();
                            self.consume_optional_whitespace();

                            let else_block = self.parse_expression()?;
                            let else_block_tipe = else_block.tipe();

                            if if_block_tipe == else_block_tipe {
                                Ok(AST::new_if_else_statement(cond, if_block, else_block, if_block_tipe, &token_info))
                            }else{

                                let token_info = else_block.token_info();
                                Err(ParseError::TypeError(format!( "Type Error: type mismatched of if and else block found at line {}, column {}",
                                token_info.line_number,
                                token_info.column_number
                            )))
                            }
                    }
                    Some(other_token) =>
                                    Err(ParseError::SyntaxError(format!( "Syntax Error: Cannot parse matching else block at line {}, column {}, found {} instead",
                                other_token.get_token_info().line_number,
                                other_token.get_token_info().column_number,
                                other_token.value(),
                            ))),
                                    _ => unreachable!(),
                }
                } else {
                    let token_info = cond.token_info();
                    Err(ParseError::TypeError(format!(
                        "Type Error: Type mismatched found at line {}, column {}",
                        token_info.line_number, token_info.column_number
                    )))
                }
            }
            Some(token) => Err(ParseError::StructureUnimplementedError(format!(
                "Parsing Error: Cannot parse ifelse statement found at line {}, column {}",
                token.get_token_info().line_number,
                token.get_token_info().column_number,
            ))),

            _ => unreachable!(),
        }
    }

    fn parse_logical_term(&mut self) -> Result<AST, ParseError> {
        self.consume_optional_whitespace();
        let mut first_expression = self.parse_comparison_term()?;
        loop {
            match self.peek() {
                Some(Token::OperatorToken {
                    operator: operator @ (Operators::LOGICAL_AND | Operators::LOGICAL_OR),
                    token_info: _,
                }) if first_expression.tipe() == LanguageType::Boolean => {
                    self.advance();
                    let second_expression = self.parse_comparison_term()?;
                    if first_expression.tipe() == second_expression.tipe() {
                        let expr = AST::new_binary_expression(
                            first_expression,
                            second_expression,
                            operator,
                        );
                        first_expression = expr;
                        continue;
                    } else {
                        let token_info = second_expression.token_info();
                        return Err(ParseError::TypeError(format!(
                            "Type Error: Type mismatched at line {}, column {}",
                            token_info.line_number, token_info.column_number
                        )));
                    }
                }
                Some(Token::OperatorToken {
                    operator: Operators::LOGICAL_AND | Operators::LOGICAL_OR,
                    token_info,
                }) if first_expression.tipe() == LanguageType::Number => {
                    return Err(ParseError::TypeError(format!(
                        "Type Error: Operator mismatched at line {}, column {}",
                        token_info.line_number, token_info.column_number
                    )));
                }
                Some(Token::WhiteSpaceToken { token_info: _ }) => {
                    self.advance();
                    continue;
                }
                _ => return Ok(first_expression),
            }
        }
    }
    fn parse_comparison_term(&mut self) -> Result<AST, ParseError> {
        let mut first_expression = self.parse_term()?;

        loop {
            match self.peek() {
                Some(Token::OperatorToken {
                    operator:
                        operator
                        @ (Operators::LESS_THAN | Operators::GREATER_THAN | Operators::EQUAL_TO),
                    token_info: _,
                }) if first_expression.tipe() == LanguageType::Number => {
                    self.advance();
                    let second_expression = self.parse_term()?;
                    if first_expression.tipe() == second_expression.tipe() {
                        let expr = AST::new_binary_expression(
                            first_expression,
                            second_expression,
                            operator,
                        );
                        first_expression = expr;
                        continue;
                    } else {
                        let token_info = second_expression.token_info();
                        return Err(ParseError::TypeError(format!(
                            "Type Error: Type mismatched at line {}, column {}",
                            token_info.line_number, token_info.column_number
                        )));
                    }
                }
                Some(Token::OperatorToken {
                    operator: operator @ Operators::EQUAL_TO,
                    token_info: _,
                }) if first_expression.tipe() == LanguageType::Boolean => {
                    self.advance();
                    let second_expression = self.parse_term()?;
                    if first_expression.tipe() == second_expression.tipe() {
                        let expr = AST::new_binary_expression(
                            first_expression,
                            second_expression,
                            operator,
                        );
                        first_expression = expr;
                        continue;
                    } else {
                        let token_info = second_expression.token_info();
                        return Err(ParseError::TypeError(format!(
                            "Type Error: Type mismatched at line {}, column {}",
                            token_info.line_number, token_info.column_number
                        )));
                    }
                }
                Some(Token::OperatorToken {
                    operator: Operators::LESS_THAN | Operators::GREATER_THAN,
                    token_info,
                }) if first_expression.tipe() == LanguageType::Boolean => {
                    return Err(ParseError::TypeError(format!(
                        "Type Error: Operator mismatched at line {}, column {}",
                        token_info.line_number, token_info.column_number
                    )));
                }
                Some(Token::WhiteSpaceToken { token_info: _ }) => {
                    self.advance();
                    continue;
                }
                _ => return Ok(first_expression),
            }
        }
    }

    fn parse_term(&mut self) -> Result<AST, ParseError> {
        let mut first_expression = self.parse_prod()?;
        loop {
            match self.peek() {
                Some(Token::OperatorToken {
                    operator: operator @ (Operators::PLUS | Operators::MINUS),
                    token_info: _,
                }) if first_expression.tipe() == LanguageType::Number => {
                    self.advance();
                    let second_expression = self.parse_prod()?;
                    let token_info = second_expression.token_info();
                    if first_expression.tipe() == second_expression.tipe() {
                        let expr = AST::new_binary_expression(
                            first_expression,
                            second_expression,
                            operator,
                        );
                        first_expression = expr;
                        continue;
                    } else {
                        return Err(ParseError::TypeError(format!(
                            "Type Error: Type mismatched at line {}, column {}",
                            token_info.line_number, token_info.column_number
                        )));
                    }
                }
                Some(Token::OperatorToken {
                    operator: Operators::PLUS | Operators::MINUS,
                    token_info,
                }) if first_expression.tipe() == LanguageType::Boolean
                    || first_expression.tipe() == LanguageType::Void =>
                {
                    return Err(ParseError::TypeError(format!(
                        "Type Error: Operator mismatched at line {}, column {}",
                        token_info.line_number, token_info.column_number
                    )));
                }
                Some(Token::WhiteSpaceToken { token_info: _ }) => {
                    self.advance();
                    continue;
                }
                _ => return Ok(first_expression),
            }
        }
    }

    fn parse_prod(&mut self) -> Result<AST, ParseError> {
        let mut first_expression = self.parse_unary()?;
        loop {
            match self.peek() {
                Some(Token::OperatorToken {
                    operator: operator @ (Operators::STAR | Operators::DIVIDE),
                    token_info: _,
                }) if first_expression.tipe() == LanguageType::Number => {
                    self.advance();
                    let second_expression = self.parse_unary()?;
                    let token_info = second_expression.token_info();
                    if first_expression.tipe() == second_expression.tipe() {
                        let expr = AST::new_binary_expression(
                            first_expression,
                            second_expression,
                            operator,
                        );
                        first_expression = expr;
                        continue;
                    } else {
                        return Err(ParseError::TypeError(format!(
                            "Type Error: Type mismatched at line {}, column {}",
                            token_info.line_number, token_info.column_number
                        )));
                    }
                }
                Some(Token::OperatorToken {
                    operator: Operators::STAR | Operators::DIVIDE,
                    token_info,
                }) if first_expression.tipe() == LanguageType::Boolean
                    || first_expression.tipe() == LanguageType::Void =>
                {
                    return Err(ParseError::TypeError(format!(
                        "Type Error: Operator mismatched at line {}, column {}",
                        token_info.line_number, token_info.column_number
                    )));
                }

                Some(Token::WhiteSpaceToken { token_info: _ }) => {
                    self.advance();
                    continue;
                }
                _ => return Ok(first_expression),
            }
        }
    }

    fn parse_unary(&mut self) -> Result<AST, ParseError> {
        match self.peek() {
            Some(Token::OperatorToken {
                operator: operator @ (Operators::PLUS | Operators::MINUS),
                token_info,
            }) => {
                self.advance();
                let val = self.parse_unary()?;
                let tipe = val.tipe();
                if tipe == LanguageType::Number {
                    let unary_expr = AST::new_unary_expression(val, operator, &token_info);
                    Ok(unary_expr)
                } else {
                    let token_info = val.token_info();
                    Err(ParseError::TypeError(format!(
                        "Type Error: Type mismatched at line {}, column {}",
                        token_info.line_number, token_info.column_number
                    )))
                }
            }
            Some(Token::OperatorToken {
                operator: operator @ Operators::LOGICAL_NOT,
                token_info,
            }) => {
                self.advance();
                let val = self.parse_unary()?;
                let tipe = val.tipe();
                if tipe == LanguageType::Boolean {
                    let unary_expr = AST::new_unary_expression(val, operator, &token_info);
                    Ok(unary_expr)
                } else {
                    let token_info = val.token_info();
                    Err(ParseError::TypeError(format!(
                        "Type Error: Type mismatched at line {}, column {}",
                        token_info.line_number, token_info.column_number
                    )))
                }
            }
            Some(Token::WhiteSpaceToken { token_info: _ }) => {
                self.advance();
                self.parse_unary()
            }
            _ => self.parse_bracket(),
        }
    }

    fn parse_bracket(&mut self) -> Result<AST, ParseError> {
        match self.peek() {
            Some(Token::SymbolToken {
                symbol: Symbols::SmallBracketOpen,
                token_info: _,
            }) => {
                self.advance();
                let val = self.parse_expression()?;
                match self.peek() {
                    Some(Token::SymbolToken {
                        symbol: Symbols::SmallBracketClose,
                        token_info: _,
                    }) => {
                        self.advance();
                        Ok(val)
                    }
                    Some(Token::WhiteSpaceToken { token_info: _ }) => {
                        self.advance();
                        self.parse_bracket()
                    }
                    Some(token) => Err(ParseError::StructureUnimplementedError(format!(
                        "Parsing Error: No closing bracket found at line {}, column {}",
                        token.get_token_info().line_number,
                        token.get_token_info().column_number
                    ))),
                    None => Err(ParseError::StructureUnimplementedError(format!(
                        "Parsing Error: None value encountered"
                    ))),
                }
            }
            _ => self.parse_block_expression(),
        }
    }

    fn parse_block_expression(&mut self) -> Result<AST, ParseError> {
        match self.peek() {
            Some(Token::SymbolToken {
                symbol: Symbols::CurlyBracketOpen,
                token_info: _,
            }) => {
                // println!("debug curly: {:?}", "here");
                self.increment_block_position();

                self.advance();
                self.consume_optional_whitespace();
                let mut statements: Vec<AST> = Vec::new();
                let mut last_statement_semicolon = false;
                loop {
                    match self.peek() {
                        Some(Token::SymbolToken {
                            symbol: Symbols::CurlyBracketClose,
                            token_info,
                        }) => {
                            self.decrement_block_position();

                            self.advance();
                            self.consume_optional_semicolon();
                            match statements.last() {
                                Some(last_statement) if !last_statement_semicolon => {
                                    let tipe = last_statement.tipe();
                                    return Ok(AST::BlockStatement {
                                        statements,
                                        tipe,
                                        token_info,
                                    });
                                }
                                Some(_) | None => {
                                    return Ok(AST::BlockStatement {
                                        statements,
                                        tipe: LanguageType::Void,
                                        token_info,
                                    });
                                }
                            }
                        }
                        Some(_) => {
                            let statement = self.parse_next_statement()?;
                            last_statement_semicolon = self.consume_optional_semicolon();
                            self.consume_optional_whitespace();
                            // println!("debug curly: {:?} {:?}", statement, self.peek());
                            statements.push(statement);
                        }
                        None => {
                            return Err(ParseError::TypeError(format!("None encountered")));
                        }
                    }
                }
            }
            _ => self.parse_identifier_expression(),
        }
    }

    fn parse_identifier_expression(&mut self) -> Result<AST, ParseError> {
        match self.peek() {
            Some(Token::IdentifierToken { lexeme, token_info }) => {
                // println!("debug in parse_literal {:?}", var_name);
                self.advance();
                let identifier_name = lexeme;
                match self.get_reference_to_identifier(&identifier_name) {
                    Some(identifier) => Ok(identifier),
                    None => Err(ParseError::IdentifierError(format!(
                "Identifier Error: No variable named '{}' found in scope at line {}, column {}",
                identifier_name,
                token_info.line_number,
                token_info.column_number,
            ))),
                }
            }
            _ => self.parse_number_literal(),
        }
    }
    fn parse_number_literal(&mut self) -> Result<AST, ParseError> {
        match self.peek() {
            Some(Token::DigitToken { lexeme, token_info }) => {
                let mut digits = vec![lexeme];
                let mut num_of_dot = 0;
                let mut digit_token_info = token_info;
                self.advance();
                loop {
                    match self.peek() {
                        Some(Token::SymbolToken {
                            symbol: Symbols::Dot,
                            token_info: _,
                        }) if num_of_dot < 1 => {
                            self.advance();
                            num_of_dot += 1;
                            digits.push(".".into());
                        }
                        Some(Token::SymbolToken {
                            symbol: Symbols::Dot,
                            token_info,
                        }) if num_of_dot >= 1 => {
                            return Err(ParseError::SyntaxError(format!(
                                "Syntax Error: Extra decimal(.) encountered at line {}, column {}",
                                token_info.line_number, token_info.column_number,
                            )));
                        }
                        Some(Token::DigitToken { lexeme, token_info }) => {
                            self.advance();
                            digit_token_info = token_info;
                            digits.push(lexeme);
                        }

                        _ => break,
                    }
                }
                Ok(AST::Number {
                    val: digits
                        .into_iter()
                        .map(|d| d.to_string())
                        .collect::<String>()
                        .parse::<f64>()
                        .unwrap(),
                    token_info: digit_token_info,
                })
            }
            _ => self.parse_literal(),
        }
    }

    fn parse_literal(&mut self) -> Result<AST, ParseError> {
        match self.peek() {
            Some(Token::KeywordToken { keyword: Keywords::True, token_info }) => {
                self.advance();
                Ok(AST::Bool{val: true, token_info})
            }
            Some(Token::KeywordToken { keyword: Keywords::False, token_info }) => {
                self.advance();
                Ok(AST::Bool{val: false, token_info})
            }

            Some(token) => Err(ParseError::SyntaxError(format!(
                "Syntax Error: Value other than literal (Number, Boolean...) '{}' encountered at line {}, column {}",
                token.value(),
                token.get_token_info().line_number,
                token.get_token_info().column_number,
            ))),

            None => unreachable!(),
        }
    }
}
#[derive(Debug, PartialEq, Eq, Copy, Clone)]
enum LanguageType {
    Number,
    Boolean,
    Void,

    PartiallyTyped,
    Untyped,
}
#[derive(Debug)]
enum AST {
    WhileStatement {
        cond: Box<AST>,
        block: Box<AST>,
        token_info: Rc<TokenInfo>,
    },

    LetStatement {
        lvalue: Rc<str>,
        rvalue: Box<AST>,
        token_info: Rc<TokenInfo>,
    },

    AssignmentStatement {
        lvalue: Rc<str>,
        rvalue: Box<AST>,
        token_info: Rc<TokenInfo>,
    },

    IfElseStatement {
        cond: Box<AST>,
        if_block: Box<AST>,
        else_block: Box<AST>,
        tipe: LanguageType,
        token_info: Rc<TokenInfo>,
    },
    BinaryExpression {
        left: Box<AST>,
        right: Box<AST>,
        operator: Operators,
    },
    UnaryExpression {
        val: Box<AST>,
        operator: Operators,
        token_info: Rc<TokenInfo>,
    },

    BlockStatement {
        statements: Vec<AST>,
        tipe: LanguageType,
        token_info: Rc<TokenInfo>,
    },
    Identifier {
        name: Rc<str>,
        block_position: usize,
        tipe: LanguageType,
        token_info: Rc<TokenInfo>,
    },

    Number {
        val: f64,
        token_info: Rc<TokenInfo>,
    },
    Bool {
        val: bool,
        token_info: Rc<TokenInfo>,
    },

    Placeholder(LanguageType),
    HaltStatement,
}
impl AST {
    fn tipe(&self) -> LanguageType {
        match self {
            Self::Number {
                val: _,
                token_info: _,
            } => LanguageType::Number,
            Self::Bool {
                val: _,
                token_info: _,
            } => LanguageType::Boolean,
            Self::UnaryExpression {
                val: _,
                operator,
                token_info: _,
            } => *operator.tipe(),
            Self::BinaryExpression {
                left: _,
                right: _,
                operator,
            } => *operator.tipe(),

            Self::Identifier {
                name: _,
                block_position: _,
                tipe,
                token_info: _,
            } => *tipe,
            Self::BlockStatement {
                statements: _,
                tipe,
                token_info: _,
            } => *tipe,

            Self::IfElseStatement {
                cond: _,
                if_block: _,
                else_block: _,
                tipe,
                token_info: _,
            } => *tipe,
            Self::Placeholder(tipe) => *tipe,

            Self::WhileStatement {
                cond: _,
                block: _,
                token_info: _,
            } => LanguageType::Void,
            Self::LetStatement {
                lvalue: _,
                rvalue: _,
                token_info: _,
            } => LanguageType::Void,
            Self::AssignmentStatement {
                lvalue: _,
                rvalue: _,
                token_info: _,
            } => LanguageType::Void,
            Self::HaltStatement => LanguageType::Void,
        }
    }
    fn token_info(&self) -> &Rc<TokenInfo> {
        match self {
            Self::Number { val: _, token_info } => token_info,
            Self::Bool { val: _, token_info } => token_info,
            Self::UnaryExpression {
                val: _,
                operator: _,
                token_info,
            } => token_info,
            Self::BinaryExpression {
                left,
                right: _,
                operator: _,
            } => left.token_info(),

            Self::Identifier {
                name: _,
                block_position: _,
                tipe: _,
                token_info,
            } => token_info,
            Self::BlockStatement {
                statements: _,
                tipe: _,
                token_info,
            } => token_info,

            Self::IfElseStatement {
                cond: _,
                if_block: _,
                else_block: _,
                tipe: _,
                token_info,
            } => token_info,

            Self::WhileStatement {
                cond: _,
                block: _,
                token_info,
            } => token_info,
            Self::LetStatement {
                lvalue: _,
                rvalue: _,
                token_info,
            } => token_info,
            Self::AssignmentStatement {
                lvalue: _,
                rvalue: _,
                token_info,
            } => token_info,

            Self::Placeholder(_) => unreachable!(),
            Self::HaltStatement => unreachable!(),
        }
    }

    fn new_binary_expression(left: Self, right: Self, operator: Operators) -> Self {
        AST::BinaryExpression {
            left: Box::new(left),
            right: Box::new(right),
            operator,
        }
    }
    fn new_unary_expression(val: Self, operator: Operators, token_info: &Rc<TokenInfo>) -> Self {
        AST::UnaryExpression {
            val: Box::new(val),
            operator,
            token_info: token_info.clone(),
        }
    }
    fn new_if_else_statement(
        cond: Self,
        if_block: Self,
        else_block: Self,
        tipe: LanguageType,
        token_info: &Rc<TokenInfo>,
    ) -> Self {
        AST::IfElseStatement {
            cond: Box::new(cond),
            if_block: Box::new(if_block),
            else_block: Box::new(else_block),
            tipe,
            token_info: token_info.clone(),
        }
    }
    fn new_assignment_statement(lvalue: &Rc<str>, rvalue: AST, token_info: &Rc<TokenInfo>) -> Self {
        AST::AssignmentStatement {
            lvalue: lvalue.clone(),
            rvalue: Box::new(rvalue),
            token_info: token_info.clone(),
        }
    }

    fn new_let_statement(lvalue: &Rc<str>, rvalue: AST, token_info: &Rc<TokenInfo>) -> Self {
        AST::LetStatement {
            lvalue: lvalue.clone(),
            rvalue: Box::new(rvalue),
            token_info: token_info.clone(),
        }
    }

    fn new_unassigned_let_statement(lvalue: &Rc<str>, token_info: &Rc<TokenInfo>) -> Self {
        AST::LetStatement {
            lvalue: lvalue.clone(),
            rvalue: Box::new(AST::Placeholder(LanguageType::Untyped)),
            token_info: token_info.clone(),
        }
    }

    fn new_while_statement(cond: Self, block: Self, token_info: &Rc<TokenInfo>) -> Self {
        AST::WhileStatement {
            cond: Box::new(cond),
            block: Box::new(block),
            token_info: token_info.clone(),
        }
    }

    fn new_identifier(
        name: &Rc<str>,
        block_position: usize,
        tipe: LanguageType,
        token_info: &Rc<TokenInfo>,
    ) -> Self {
        AST::Identifier {
            name: name.clone(),
            block_position,
            tipe,
            token_info: token_info.clone(),
        }
    }

    fn get_identifier_name(&self) -> Option<Rc<str>> {
        match self {
            Self::Identifier {
                name,
                block_position: _,
                tipe: _,
                token_info: _,
            } => Some(name.clone()),
            _ => None,
        }
    }
    fn clone_identifier(&self) -> Option<Self> {
        match self {
            Self::Identifier {
                name,
                block_position,
                tipe,
                token_info,
            } => Some(AST::new_identifier(
                name,
                *block_position,
                *tipe,
                token_info,
            )),
            _ => None,
        }
    }
    fn clone_and_modify_identifier(&self, tipe_value: LanguageType) -> Option<Self> {
        match self {
            Self::Identifier {
                name,
                block_position,
                tipe: _,
                token_info,
            } => Some(AST::new_identifier(
                name,
                *block_position,
                tipe_value,
                token_info,
            )),
            _ => None,
        }
    }
}

#[derive(Debug)]
enum ParseError {
    StructureUnimplementedError(String),
    TypeError(String),
    IdentifierError(String),
    SyntaxError(String),
}

///////////////////////////////////////////////////////// Interpreter Code /////////////////////////////////////////////////
#[derive(Debug, Clone, Copy)]
enum InterpreterDataStucture {
    Number(f64),
    Bool(bool),
    Void,
}

impl InterpreterDataStucture {
    fn match_binary_operation(left: Self, right: Self, operator: Operators) -> Self {
        match (left, right) {
            (Self::Number(l), Self::Number(r)) => match operator {
                Operators::PLUS => Self::Number(l + r),
                Operators::MINUS => Self::Number(l - r),
                Operators::STAR => Self::Number(l * r),
                Operators::DIVIDE => Self::Number(l / r),

                Operators::LESS_THAN => Self::Bool(l < r),
                Operators::GREATER_THAN => Self::Bool(l > r),
                Operators::EQUAL_TO => Self::Bool(l == r),

                _ => {
                    println!("{:?}, {:?}", left, right);
                    unreachable!();
                }
            },

            (Self::Bool(l), Self::Bool(r)) => match operator {
                Operators::LOGICAL_AND => Self::Bool(l && r),
                Operators::LOGICAL_OR => Self::Bool(l || r),
                Operators::EQUAL_TO => Self::Bool(l == r),
                _ => {
                    println!("{:?}, {:?}", left, right);
                    unreachable!()
                }
            },

            _ => {
                println!("{:?}, {:?}", left, right);
                unreachable!()
            }
        }
    }
    fn match_unary_operation(val: Self, operator: Operators) -> Self {
        match val {
            Self::Number(v) => match operator {
                Operators::PLUS => Self::Number(v),
                Operators::MINUS => Self::Number(-v),
                _ => unreachable!(),
            },
            Self::Bool(b) => match operator {
                Operators::LOGICAL_NOT => Self::Bool(!b),
                _ => unreachable!(),
            },

            _ => unreachable!(),
        }
    }
}
#[derive(Debug)]
struct Environment {
    container: Vec<HashMap<Rc<str>, InterpreterDataStucture>>,
}

impl Environment {
    fn new() -> Self {
        Self {
            container: vec![HashMap::new()],
        }
    }
    fn insert_new(&mut self, lvalue: Rc<str>, rvalue: InterpreterDataStucture) {
        self.container.last_mut().unwrap().insert(lvalue, rvalue);
    }
    fn insert(&mut self, lvalue: Rc<str>, rvalue: InterpreterDataStucture) {
        let mut location = 0;
        for (i, child_container) in self.container.iter().enumerate().rev() {
            if child_container.contains_key(&lvalue) {
                location = i;
            }
        }
        self.container[location].insert(lvalue, rvalue);
    }
    fn get(&self, name: &str) -> Option<&InterpreterDataStucture> {
        for container in self.container.iter().rev() {
            if container.contains_key(name) {
                return container.get(name);
            }
        }
        None
    }
    fn create_child(&mut self) {
        self.container.push(HashMap::new());
    }
    fn pop_child(&mut self) {
        self.container.pop();
    }
}
struct Interpreter {
    environment: Environment,
    previous_identifiers: Option<Vec<AST>>,
}

impl Interpreter {
    fn new() -> Self {
        Interpreter {
            environment: Environment::new(),
            previous_identifiers: None,
        }
    }

    fn calculate_statement(&mut self, statement: &AST) -> InterpreterDataStucture {
        match statement {
            AST::Number { val, token_info: _ } => InterpreterDataStucture::Number(*val),

            AST::Bool { val, token_info: _ } => InterpreterDataStucture::Bool(*val),

            AST::BinaryExpression {
                left,
                right,
                operator,
            } => InterpreterDataStucture::match_binary_operation(
                self.calculate_statement(&left),
                self.calculate_statement(&right),
                *operator,
            ),

            AST::UnaryExpression {
                val,
                operator,
                token_info: _,
            } => InterpreterDataStucture::match_unary_operation(
                self.calculate_statement(&val),
                *operator,
            ),
            AST::LetStatement {
                lvalue,
                rvalue,
                token_info: _,
            } => {
                let return_val = self.calculate_statement(&rvalue);
                self.environment.insert_new(lvalue.clone(), return_val);
                InterpreterDataStucture::Void
            }
            AST::AssignmentStatement {
                lvalue,
                rvalue,
                token_info: _,
            } => {
                let return_val = self.calculate_statement(&rvalue);
                self.environment.insert(lvalue.clone(), return_val);
                InterpreterDataStucture::Void
            }
            AST::Identifier {
                name,
                block_position: _,
                tipe: _,
                token_info: _,
            } => self.environment.get(name).unwrap().clone(),

            AST::BlockStatement {
                statements,
                tipe,
                token_info: _,
            } => {
                let mut result = InterpreterDataStucture::Void;
                self.environment.create_child();
                for statement in statements {
                    result = self.calculate_statement(statement);
                }
                self.environment.pop_child();
                match tipe {
                    LanguageType::Void => InterpreterDataStucture::Void,
                    _ => result,
                }
            }

            AST::WhileStatement {
                cond,
                block,
                token_info: _,
            } => {
                let mut result = InterpreterDataStucture::Void;
                loop {
                    match self.calculate_statement(cond) {
                        InterpreterDataStucture::Bool(b) if b == true => {
                            result = self.calculate_statement(block);
                        }
                        InterpreterDataStucture::Bool(b) if b == false => break,
                        _ => unreachable!(),
                    }
                }
                result
            }
            AST::IfElseStatement {
                cond,
                if_block,
                else_block,
                tipe: _,
                token_info: _,
            } => {
                let result;
                match self.calculate_statement(cond) {
                    InterpreterDataStucture::Bool(b) if b == true => {
                        result = self.calculate_statement(if_block);
                    }
                    InterpreterDataStucture::Bool(b) if b == false => {
                        result = self.calculate_statement(else_block);
                    }
                    _ => unreachable!(),
                }
                result
            }
            AST::Placeholder(_) => InterpreterDataStucture::Void,
            AST::HaltStatement => InterpreterDataStucture::Void,
        }
    }

    fn tokenize_and_parse(
        &mut self,
        input: &str,
        previous_identifiers: Option<Vec<AST>>,
    ) -> Result<(Vec<AST>, Option<Vec<AST>>), ParseError> {
        let mut tokenizer = Tokenizer::new(&input);
        let tokens = tokenizer.tokenize();

        match tokens {
            Ok(token_list) => {
                let mut parser = Parser::new(&token_list);
                parser.set_previous_identifiers_list(previous_identifiers);

                match parser.parse() {
                    Ok(parsed_statements) => Ok((parsed_statements, parser.get_identifiers_list())),
                    Err(err) => {
                        self.set_previous_identifiers(parser.get_identifiers_list());
                        Err(err)
                    }
                }
            }
            Err(err) => {
                self.set_previous_identifiers(previous_identifiers);
                Err(ParseError::StructureUnimplementedError(
                    err.into_iter()
                        .map(|x| x + "\n")
                        .collect::<String>()
                        .trim_end()
                        .to_string(),
                ))
            }
        }
    }
    fn set_previous_identifiers(&mut self, previous_identifiers: Option<Vec<AST>>) {
        self.previous_identifiers = previous_identifiers;
    }
    fn interpret(&mut self, input: &str) {
        let previous_identifiers = self.previous_identifiers.take();
        match self.tokenize_and_parse(input, previous_identifiers) {
            Ok((parsed_statements, new_identifiers)) => {
                for statement in parsed_statements {
                    match self.calculate_statement(&statement) {
                        InterpreterDataStucture::Void => continue,
                        result => println!("{:?}", result),
                    }
                }
                self.set_previous_identifiers(new_identifiers);
            }
            Err(error) => {
                println!("{:?}", error);
            }
        }
        // println!("{:?}", self.environment);
    }

    fn interactive_prompt(&mut self) {
        'loop_start: loop {
            let mut line = String::new();
            print!(">>> ");

            let _ = std::io::stdout().flush();
            let _ = std::io::stdin().read_line(&mut line).unwrap();

            let line = line.trim_end();

            if line.len() == 0 {
                continue 'loop_start;
            }
            self.interpret(&line);
        }
    }
}
