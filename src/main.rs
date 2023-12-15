use std::{
    collections::HashMap, format, io::Write, println, rc::Rc, unimplemented, unreachable, vec,
};

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
    fn is_equal<'a>(&self, check_with: &'a [char]) -> bool {
        check_with.iter().collect::<String>() == self.lexeme()
    }
}

#[derive(PartialEq, Debug, Clone)]
enum Token {
    DigitToken {
        lexeme: Rc<str>,
        token_info: TokenInfo,
    },
    AlphabetToken {
        lexeme: Rc<str>,
        token_info: TokenInfo,
    },
    IdentifierToken {
        lexeme: Rc<str>,
        token_info: TokenInfo,
    },
    OperatorToken {
        operator: Operators,
        token_info: TokenInfo,
    },
    SymbolToken {
        symbol: Symbols,
        token_info: TokenInfo,
    },
    KeywordToken {
        keyword: Keywords,
        token_info: TokenInfo,
    },

    WhiteSpaceToken {
        token_info: TokenInfo,
    },
    NewLineToken {
        token_info: TokenInfo,
    },
    EOFToken {
        token_info: TokenInfo,
    },
}
impl Token {
    fn get_token_info(&self) -> &TokenInfo {
        match self {
            Self::DigitToken { lexeme, token_info } => &token_info,
            Self::AlphabetToken { lexeme, token_info } => &token_info,
            Self::IdentifierToken { lexeme, token_info } => &token_info,

            Self::OperatorToken {
                operator,
                token_info,
            } => &token_info,
            Self::SymbolToken { symbol, token_info } => &token_info,
            Self::KeywordToken {
                keyword,
                token_info,
            } => &token_info,

            Self::WhiteSpaceToken { token_info } => &token_info,
            Self::NewLineToken { token_info } => &token_info,

            Self::EOFToken { token_info } => &token_info,
        }
    }
    fn value(&self) -> Rc<str> {
        match self {
            Self::DigitToken { lexeme, token_info } => lexeme.clone(),
            Self::AlphabetToken { lexeme, token_info } => lexeme.clone(),
            Self::IdentifierToken { lexeme, token_info } => lexeme.clone(),

            Self::OperatorToken {
                operator,
                token_info,
            } => operator.lexeme().into(),

            Self::SymbolToken { symbol, token_info } => symbol.lexeme().into(),
            Self::KeywordToken {
                keyword,
                token_info,
            } => keyword.lexeme().into(),

            Self::WhiteSpaceToken { token_info } => " ".into(),
            Self::NewLineToken { token_info } => "\n".into(),

            Self::EOFToken { token_info } => "<EOF>".into(),
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
    identifiers_list: Option<Vec<Identifier>>,
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

    fn get_identifiers_list(self) -> Option<Vec<Identifier>> {
        self.identifiers_list
    }

    fn push_identifier(&mut self, identifier: Identifier) {
        if self.identifiers_list.is_none() {
            self.identifiers_list = Some(vec![identifier]);
        } else {
            self.identifiers_list.as_mut().unwrap().push(identifier);
        }

        self.increment_num_identifier_in_current_block();
    }

    fn set_previous_identifiers_list(&mut self, previous_identifier_list: Option<Vec<Identifier>>) {
        self.identifiers_list = previous_identifier_list
    }

    fn get_reference_to_identifier(&self, name: &str) -> Option<Identifier> {
        match &self.identifiers_list {
            Some(identifiers) => {
                for identifier in identifiers.iter().rev() {
                    if name == identifier.name {
                        return Some(identifier.clone());
                    }
                }
            }
            None => return None,
        }
        None
    }

    fn modify_identifier_type(
        &mut self,
        identifier_to_modify: &Identifier,
        tipe_value: LanguageType,
    ) {
        match self.identifiers_list.as_deref_mut() {
            Some(identifiers) => {
                for identifier in identifiers.iter_mut().rev() {
                    if identifier.name == identifier_to_modify.name {
                        *identifier = Identifier::new(
                            &identifier.name,
                            identifier.block_position,
                            tipe_value,
                        );
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

    fn peek(&self) -> Option<&Token> {
        if self.index < self.len {
            Some(&self.tokens[self.index])
        } else {
            None
        }
    }

    fn consume_optional_semicolon(&mut self) -> bool {
        match self.peek() {
            Some(Token::SymbolToken {
                symbol: Symbols::SemiColon,
                token_info,
            }) => {
                self.advance();
                true
            }
            None | Some(_) => false,
        }
    }

    fn consume_optional_whitespace(&mut self) -> bool {
        match self.peek() {
            Some(Token::WhiteSpaceToken { token_info }) => {
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
            .map(|(statement, _)| statement)
    }
    fn parse_halt_statement(&mut self) -> Result<(AST, TokenInfo), ParseError> {
        match self.peek().cloned() {
            Some(Token::EOFToken { token_info }) => {
                self.advance();
                Ok((AST::HaltStatement, token_info))
            }
            Some(token) => Err(ParseError::StructureUnimplementedError(format!(
                "Parsing Error: Cannot parse halt statement found at line {}, column {}",
                token.get_token_info().line_number,
                token.get_token_info().column_number,
            ))),
            None => unreachable!(),
        }
    }
    fn parse_while_statement(&mut self) -> Result<(AST, TokenInfo), ParseError> {
        self.consume_optional_whitespace();
        match self.peek().cloned() {
            Some(Token::KeywordToken {
                keyword: Keywords::While,
                token_info,
            }) => {
                self.advance();
                self.consume_optional_whitespace();
                let (condition_expression, token_info) = self.parse_logical_term()?;
                if condition_expression.tipe() == &LanguageType::Boolean {
                    self.consume_optional_whitespace();
                    let (block_expression, _) = self.parse_logical_term()?;
                    let tipe = block_expression.tipe().clone();
                    Ok((
                        AST::WhileStatement(
                            Box::new(condition_expression),
                            Box::new(block_expression),
                        ),
                        token_info,
                    ))
                } else {
                    Err(ParseError::StructureUnimplementedError(format!(
                        "Type Error: Cannot parse while statement found at line {}, column {}",
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

    fn parse_let_statement(&mut self) -> Result<(AST, TokenInfo), ParseError> {
        self.consume_optional_whitespace();
        match self.peek().cloned() {
            Some(Token::KeywordToken {
                keyword: Keywords::Let,
                token_info,
            }) => {
                self.advance();
                self.advance(); //consume whitespace
                match self.peek().cloned() {
                    Some(Token::IdentifierToken { lexeme, token_info }) => {
                        self.advance();
                        self.consume_optional_whitespace();
                        match self.peek().cloned() {
                            Some(Token::SymbolToken {
                                symbol: Symbols::Equal,
                                token_info,
                            }) => {
                                self.advance();
                                self.consume_optional_whitespace();
                                let (term, token_info) = self.parse_expression()?;

                                let tipe = term.tipe();

                                match tipe {
                                    LanguageType::Void => Err(ParseError::TypeError(format!(
                                        "Type Error: Void type found at line {}, column {}",
                                        token_info.line_number, token_info.column_number
                                    ))),
                                    _ => {
                                        self.push_identifier(Identifier::new(
                                            &lexeme,
                                            self.get_block_position(),
                                            *tipe,
                                        ));

                                        Ok((
                                            LetStatement::new(lexeme.to_string(), term),
                                            token_info,
                                        ))
                                    }
                                }
                            }
                            Some(token) => {
                                self.push_identifier(Identifier::new(
                                    &lexeme,
                                    self.get_block_position(),
                                    LanguageType::Void,
                                ));

                                Ok((
                                    LetStatement::new(
                                        lexeme.to_string(),
                                        AST::Placeholder(LanguageType::Untyped),
                                    ),
                                    token_info,
                                ))
                            }
                            Some(token) => Err(ParseError::SyntaxError(format!(
                                "Syntax Error: No equal found at line {}, column {}",
                                token.get_token_info().line_number,
                                token.get_token_info().column_number
                            ))),
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
    fn parse_assignment_statement(&mut self) -> Result<(AST, TokenInfo), ParseError> {
        self.consume_optional_whitespace();
        self.remember_index();
        match self.peek().cloned() {
            Some(Token::IdentifierToken { lexeme, token_info }) => {
                match self.get_reference_to_identifier(&lexeme) {
                    Some(identifier) => {
                        self.advance();
                        self.consume_optional_whitespace();
                        match self.peek().cloned() {
                            Some(Token::SymbolToken{symbol: Symbols::Equal, token_info}) =>
                            {
                                self.advance();
                                self.consume_optional_whitespace();
                                let (term, token_info) = self.parse_expression()?;
                                if term.tipe() == identifier.tipe() || identifier.tipe() == &LanguageType::Void{
                                    self.modify_identifier_type(&identifier, *term.tipe());
                                    Ok((
                                        AssignmentStatement::new(lexeme.to_string(), term),
                                        token_info,
                                    ))
                                } else {
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

    fn parse_expression(&mut self) -> Result<(AST, TokenInfo), ParseError> {
        self.parse_if_else_expression().or_else(|e| match e {
            ParseError::StructureUnimplementedError(_) => self.parse_logical_term(),
            ParseError::TypeError(_)
            | ParseError::IdentifierError(_)
            | ParseError::SyntaxError(_) => Err(e),
        })
    }

    fn parse_if_else_expression(&mut self) -> Result<(AST, TokenInfo), ParseError> {
        self.consume_optional_whitespace();
        match self.peek().cloned() {
            Some(Token::KeywordToken {
                keyword: Keywords::If,
                token_info,
            }) => {
                self.advance();
                self.consume_optional_whitespace();
                let (condition_expression, token_info) = self.parse_logical_term()?;
                if condition_expression.tipe() == &LanguageType::Boolean {
                    self.consume_optional_whitespace();
                    let (if_block_expression, _) = self.parse_logical_term()?;
                    let if_block_tipe = if_block_expression.tipe().clone();

                    self.consume_optional_whitespace();

                    match self.peek().cloned(){
                        Some(Token::KeywordToken{keyword: Keywords::Else, token_info}) => {
                            self.advance();
                            self.consume_optional_whitespace();

                            let (else_block_expression, token_info) = self.parse_expression()?;
                            let else_block_tipe = else_block_expression.tipe();

                            if if_block_tipe == *else_block_tipe {
                                Ok((AST::IfElseStatement(Box::new(condition_expression), Box::new(if_block_expression), Box::new(else_block_expression), if_block_tipe), token_info))
                            }else{

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

    fn parse_logical_term(&mut self) -> Result<(AST, TokenInfo), ParseError> {
        self.consume_optional_whitespace();
        let (mut first_expression, token_info) = self.parse_comparison_term()?;
        loop {
            match self.peek().cloned() {
                Some(Token::OperatorToken {
                    operator: operator @ (Operators::LOGICAL_AND | Operators::LOGICAL_OR),

                    token_info,
                }) if first_expression.tipe() == &LanguageType::Boolean => {
                    self.advance();
                    let (second_expression, token_info) = self.parse_comparison_term()?;
                    if first_expression.tipe() == second_expression.tipe() {
                        let expr =
                            BinaryExpression::new(first_expression, second_expression, operator);
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
                    operator: Operators::LOGICAL_AND | Operators::LOGICAL_OR,
                    token_info,
                }) if first_expression.tipe() == &LanguageType::Number => {
                    return Err(ParseError::TypeError(format!(
                        "Type Error: Operator mismatched at line {}, column {}",
                        token_info.line_number, token_info.column_number
                    )));
                }
                Some(Token::WhiteSpaceToken { token_info }) => {
                    self.advance();
                    continue;
                }
                _ => {
                    return Ok((first_expression, token_info));
                }
            }
        }
    }
    fn parse_comparison_term(&mut self) -> Result<(AST, TokenInfo), ParseError> {
        let (mut first_expression, token_info) = self.parse_term()?;

        loop {
            match self.peek().cloned() {
                Some(Token::OperatorToken {
                    operator:
                        operator
                        @ (Operators::LESS_THAN | Operators::GREATER_THAN | Operators::EQUAL_TO),
                    token_info,
                }) if first_expression.tipe() == &LanguageType::Number => {
                    self.advance();
                    let (second_expression, token_info) = self.parse_term()?;
                    if first_expression.tipe() == second_expression.tipe() {
                        let expr =
                            BinaryExpression::new(first_expression, second_expression, operator);
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
                    operator: operator @ Operators::EQUAL_TO,
                    token_info,
                }) if first_expression.tipe() == &LanguageType::Boolean => {
                    self.advance();
                    let (second_expression, token_info) = self.parse_term()?;
                    if first_expression.tipe() == second_expression.tipe() {
                        let expr =
                            BinaryExpression::new(first_expression, second_expression, operator);
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
                    operator: operator @ (Operators::LESS_THAN | Operators::GREATER_THAN),
                    token_info,
                }) if first_expression.tipe() == &LanguageType::Boolean => {
                    return Err(ParseError::TypeError(format!(
                        "Type Error: Operator mismatched at line {}, column {}",
                        token_info.line_number, token_info.column_number
                    )));
                }
                Some(Token::WhiteSpaceToken { token_info }) => {
                    self.advance();
                    continue;
                }
                _ => return Ok((first_expression, token_info)),
            }
        }
    }

    fn parse_term(&mut self) -> Result<(AST, TokenInfo), ParseError> {
        let (mut first_expression, token_info) = self.parse_prod()?;
        loop {
            match self.peek().cloned() {
                Some(Token::OperatorToken {
                    operator: operator @ (Operators::PLUS | Operators::MINUS),
                    token_info,
                }) if first_expression.tipe() == &LanguageType::Number => {
                    self.advance();
                    let (second_expression, token_info) = self.parse_prod()?;
                    if first_expression.tipe() == second_expression.tipe() {
                        let expr =
                            BinaryExpression::new(first_expression, second_expression, operator);
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
                }) if first_expression.tipe() == &LanguageType::Boolean
                    || first_expression.tipe() == &LanguageType::Void =>
                {
                    return Err(ParseError::TypeError(format!(
                        "Type Error: Operator mismatched at line {}, column {}",
                        token_info.line_number, token_info.column_number
                    )));
                }
                Some(Token::WhiteSpaceToken { token_info }) => {
                    self.advance();
                    continue;
                }
                _ => {
                    return Ok((first_expression, token_info));
                }
            }
        }
    }

    fn parse_prod(&mut self) -> Result<(AST, TokenInfo), ParseError> {
        let (mut first_expression, token_info) = self.parse_unary()?;
        loop {
            match self.peek().cloned() {
                Some(Token::OperatorToken {
                    operator: operator @ (Operators::STAR | Operators::DIVIDE),
                    token_info,
                }) if first_expression.tipe() == &LanguageType::Number => {
                    self.advance();
                    let (second_expression, token_info) = self.parse_unary()?;
                    if first_expression.tipe() == second_expression.tipe() {
                        let expr =
                            BinaryExpression::new(first_expression, second_expression, operator);
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
                    operator: operator @ (Operators::STAR | Operators::DIVIDE),
                    token_info,
                }) if first_expression.tipe() == &LanguageType::Boolean
                    || first_expression.tipe() == &LanguageType::Void =>
                {
                    return Err(ParseError::TypeError(format!(
                        "Type Error: Operator mismatched at line {}, column {}",
                        token_info.line_number, token_info.column_number
                    )));
                }

                Some(Token::WhiteSpaceToken { token_info }) => {
                    self.advance();
                    continue;
                }
                _ => {
                    return Ok((first_expression, token_info));
                }
            }
        }
    }

    fn parse_unary(&mut self) -> Result<(AST, TokenInfo), ParseError> {
        match self.peek().cloned() {
            Some(Token::OperatorToken {
                operator: operator @ (Operators::PLUS | Operators::MINUS),
                token_info,
            }) => {
                self.advance();
                let (expr, token_info) = self.parse_unary()?;
                let tipe = expr.tipe().clone();
                let unary_expr = UnaryExpression::new(expr, operator);
                if tipe == LanguageType::Number {
                    Ok((unary_expr, token_info))
                } else {
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
                let (expr, token_info) = self.parse_unary()?;
                let tipe = expr.tipe().clone();
                let unary_expr = UnaryExpression::new(expr, operator);
                if tipe == LanguageType::Boolean {
                    Ok((unary_expr, token_info))
                } else {
                    Err(ParseError::TypeError(format!(
                        "Type Error: Type mismatched at line {}, column {}",
                        token_info.line_number, token_info.column_number
                    )))
                }
            }
            Some(Token::WhiteSpaceToken { token_info }) => {
                self.advance();
                self.parse_unary()
            }
            _ => self.parse_bracket(),
        }
    }

    fn parse_bracket(&mut self) -> Result<(AST, TokenInfo), ParseError> {
        match self.peek().cloned() {
            Some(Token::SymbolToken {
                symbol: Symbols::SmallBracketOpen,
                token_info,
            }) => {
                self.advance();
                let (val, token_info) = self.parse_expression()?;
                match self.peek().cloned() {
                    Some(Token::SymbolToken {
                        symbol: Symbols::SmallBracketClose,
                        token_info,
                    }) => {
                        self.advance();
                        Ok((val, token_info))
                    }
                    Some(Token::WhiteSpaceToken { token_info }) => {
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

    fn parse_block_expression(&mut self) -> Result<(AST, TokenInfo), ParseError> {
        match self.peek().cloned() {
            Some(Token::SymbolToken {
                symbol: Symbols::CurlyBracketOpen,
                token_info,
            }) => {
                // println!("debug curly: {:?}", "here");
                self.increment_block_position();

                self.advance();
                self.consume_optional_whitespace();
                let mut statements: Vec<AST> = Vec::new();
                let mut last_statement_semicolon = false;
                loop {
                    match self.peek().cloned() {
                        Some(Token::SymbolToken {
                            symbol: Symbols::CurlyBracketClose,
                            token_info,
                        }) => {
                            self.decrement_block_position();

                            self.advance();
                            self.consume_optional_semicolon();
                            match statements.last() {
                                Some(last_statement) if !last_statement_semicolon => {
                                    let tipe = last_statement.tipe().clone();
                                    return Ok((AST::BlockStatement(statements, tipe), token_info));
                                }
                                Some(_) => {
                                    return Ok((
                                        AST::BlockStatement(statements, LanguageType::Void),
                                        token_info,
                                    ));
                                }
                                None => {
                                    return Ok((
                                        AST::BlockStatement(statements, LanguageType::Void),
                                        token_info,
                                    ));
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

    fn parse_identifier_expression(&mut self) -> Result<(AST, TokenInfo), ParseError> {
        match self.peek().cloned() {
            Some(Token::IdentifierToken { lexeme, token_info }) => {
                // println!("debug in parse_literal {:?}", var_name);
                self.advance();
                let identifier_name = lexeme;
                match self.get_reference_to_identifier(&identifier_name) {
                    Some(identifier) => Ok((identifier.clone_to_ast(), token_info)),
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
    fn parse_number_literal(&mut self) -> Result<(AST, TokenInfo), ParseError> {
        match self.peek().cloned() {
            Some(Token::DigitToken { lexeme, token_info }) => {
                let mut digits = vec![lexeme];
                let mut num_of_dot = 0;
                let mut digit_token_info = token_info;
                self.advance();
                loop {
                    match self.peek().cloned() {
                        Some(Token::SymbolToken {
                            symbol: Symbols::Dot,
                            token_info,
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

                        _ => {
                            break;
                        }
                    }
                }
                Ok((
                    AST::Number(
                        digits
                            .into_iter()
                            .map(|d| d.to_string())
                            .collect::<String>()
                            .parse::<f64>()
                            .unwrap(),
                    ),
                    digit_token_info,
                ))
            }
            _ => self.parse_literal(),
        }
    }

    fn parse_literal(&mut self) -> Result<(AST, TokenInfo), ParseError> {
        match self.peek().cloned() {
            Some(Token::KeywordToken { keyword: Keywords::True, token_info }) => {
                self.advance();
                Ok((AST::Bool(true), token_info))
            }
            Some(Token::KeywordToken { keyword: Keywords::False, token_info }) => {
                self.advance();
                Ok((AST::Bool(false), token_info))
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
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
enum LanguageType {
    Number,
    Boolean,
    Void,

    PartiallyTyped,
    Untyped,
}
#[derive(Debug)]
enum AST {
    WhileStatement(Box<AST>, Box<AST>),
    LetStatement(Box<LetStatement>),
    AssignmentStatement(Box<AssignmentStatement>),

    IfElseStatement(Box<AST>, Box<AST>, Box<AST>, LanguageType),
    BinaryExpression(Box<BinaryExpression>),
    UnaryExpression(Box<UnaryExpression>),
    BlockStatement(Vec<AST>, LanguageType),
    Identifier(Identifier),
    Number(f64),
    Bool(bool),

    Placeholder(LanguageType),

    HaltStatement,
}

impl AST {
    fn tipe(&self) -> &LanguageType {
        match self {
            Self::Number(_) => &LanguageType::Number,
            Self::Bool(_) => &LanguageType::Boolean,
            Self::BinaryExpression(boxed_binary_expression) => boxed_binary_expression.tipe(),
            Self::UnaryExpression(boxed_unary_expression) => boxed_unary_expression.tipe(),
            Self::LetStatement(boxed_let_statement) => boxed_let_statement.tipe(),
            Self::AssignmentStatement(boxed_assignment_statement) => {
                boxed_assignment_statement.tipe()
            }
            Self::Identifier(identifier) => identifier.tipe(),
            Self::BlockStatement(_, tipe) => tipe,
            Self::WhileStatement(_, _) => &LanguageType::Void,
            Self::IfElseStatement(_, _, _, tipe) => tipe,

            Self::Placeholder(tipe) => tipe,
            Self::HaltStatement => &LanguageType::Void,
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

#[derive(Debug)]
struct BinaryExpression {
    left: AST,
    right: AST,
    operator: Operators,
}
impl BinaryExpression {
    fn new(left: AST, right: AST, operator: Operators) -> AST {
        AST::BinaryExpression(Box::new(Self {
            left: left,
            right: right,
            operator: operator,
        }))
    }
    fn tipe(&self) -> &LanguageType {
        self.operator.tipe()
    }
}
#[derive(Debug)]
struct UnaryExpression {
    val: AST,
    operator: Operators,
}
impl UnaryExpression {
    fn new(val: AST, operator: Operators) -> AST {
        AST::UnaryExpression(Box::new(Self {
            val: val,
            operator: operator,
        }))
    }
    fn tipe(&self) -> &LanguageType {
        self.operator.tipe()
    }
}

#[derive(Debug)]
struct AssignmentStatement {
    lvalue: String,
    rvalue: AST,
}

impl AssignmentStatement {
    fn new(lvalue: String, rvalue: AST) -> AST {
        AST::AssignmentStatement(Box::new(Self {
            lvalue: lvalue,
            rvalue: rvalue,
        }))
    }
    fn tipe(&self) -> &LanguageType {
        &LanguageType::Void
    }
}
#[derive(Debug)]
struct LetStatement {
    lvalue: String,
    rvalue: AST,
}

impl LetStatement {
    fn new(lvalue: String, rvalue: AST) -> AST {
        AST::LetStatement(Box::new(Self {
            lvalue: lvalue,
            rvalue: rvalue,
        }))
    }
    fn tipe(&self) -> &LanguageType {
        &LanguageType::Void
    }
}

#[derive(Debug)]
struct Identifier {
    name: String,
    block_position: usize,
    tipe: LanguageType,
}

impl Identifier {
    fn new(name: &str, block_position: usize, tipe: LanguageType) -> Self {
        Self {
            name: name.to_string(),
            block_position: block_position,
            tipe: tipe,
        }
    }
    fn new_ast(name: String, block_position: usize, tipe: LanguageType) -> AST {
        AST::Identifier(Self {
            name: name,
            block_position: block_position,
            tipe: tipe,
        })
    }
    fn tipe(&self) -> &LanguageType {
        &self.tipe
    }
    fn clone(&self) -> Identifier {
        Self::new(&self.name, self.block_position, self.tipe)
    }
    fn clone_to_ast(&self) -> AST {
        Self::new_ast(self.name.to_string(), self.block_position, self.tipe)
    }
}

///////////////////////////////////////////////////////// Interpreter Code /////////////////////////////////////////////////
#[derive(Debug, Clone, Copy)]
enum InternalDataStucture {
    Number(f64),
    Bool(bool),
    Void,
}

impl InternalDataStucture {
    fn __match_binary_operation__(left: Self, right: Self, operator: Operators) -> Self {
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
    fn __match_unary_operation__(val: Self, operator: Operators) -> Self {
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
    container: Vec<HashMap<String, InternalDataStucture>>,
}

impl Environment {
    fn new() -> Self {
        Self {
            container: vec![HashMap::new()],
        }
    }
    fn insert_new(&mut self, lvalue: String, rvalue: InternalDataStucture) {
        self.container.last_mut().unwrap().insert(lvalue, rvalue);
    }
    fn insert(&mut self, lvalue: String, rvalue: InternalDataStucture) {
        let mut location = 0;
        for (i, child_container) in self.container.iter().enumerate().rev() {
            if child_container.contains_key(&lvalue) {
                location = i;
            }
        }
        self.container[location].insert(lvalue, rvalue);
    }
    fn get(&self, name: &str) -> Option<&InternalDataStucture> {
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
    previous_identifiers: Option<Vec<Identifier>>,
}

impl Interpreter {
    fn new() -> Self {
        Interpreter {
            environment: Environment::new(),
            previous_identifiers: None,
        }
    }

    fn calculate_statement(&mut self, statement: &AST) -> InternalDataStucture {
        match statement {
            AST::Number(val) => InternalDataStucture::Number(*val),

            AST::Bool(val) => InternalDataStucture::Bool(*val),

            AST::BinaryExpression(boxed_expression) => match &**boxed_expression {
                BinaryExpression {
                    left,
                    right,
                    operator,
                } => InternalDataStucture::__match_binary_operation__(
                    self.calculate_statement(&left),
                    self.calculate_statement(&right),
                    *operator,
                ),
            },
            AST::UnaryExpression(boxed_expression) => match &**boxed_expression {
                UnaryExpression { val, operator } => {
                    InternalDataStucture::__match_unary_operation__(
                        self.calculate_statement(&val),
                        *operator,
                    )
                }
            },
            AST::LetStatement(boxed_let_statement) => match &**boxed_let_statement {
                LetStatement { lvalue, rvalue } => {
                    let return_val = self.calculate_statement(&rvalue);
                    self.environment.insert_new(lvalue.clone(), return_val);
                    InternalDataStucture::Void
                }
            },
            AST::AssignmentStatement(boxed_assignment_statement) => {
                match &**boxed_assignment_statement {
                    AssignmentStatement { lvalue, rvalue } => {
                        let return_val = self.calculate_statement(&rvalue);
                        self.environment.insert(lvalue.clone(), return_val);
                        InternalDataStucture::Void
                    }
                }
            }
            AST::Identifier(identifier) => self.environment.get(&identifier.name).unwrap().clone(),

            AST::BlockStatement(statements, tipe) => {
                let mut result = InternalDataStucture::Void;
                self.environment.create_child();
                for statement in statements {
                    result = self.calculate_statement(statement);
                }
                self.environment.pop_child();
                match tipe {
                    LanguageType::Void => InternalDataStucture::Void,
                    _ => result,
                }
            }

            AST::WhileStatement(condition, block_statement) => {
                let mut result = InternalDataStucture::Void;
                loop {
                    match self.calculate_statement(condition) {
                        InternalDataStucture::Bool(b) if b == true => {
                            result = self.calculate_statement(block_statement);
                        }
                        InternalDataStucture::Bool(b) if b == false => break,
                        _ => unreachable!(),
                    }
                }
                result
            }
            AST::IfElseStatement(condition, if_block_statement, else_block_statement, _) => {
                let result;
                match self.calculate_statement(condition) {
                    InternalDataStucture::Bool(b) if b == true => {
                        result = self.calculate_statement(if_block_statement);
                    }
                    InternalDataStucture::Bool(b) if b == false => {
                        result = self.calculate_statement(else_block_statement);
                    }
                    _ => unreachable!(),
                }
                result
            }
            AST::Placeholder(tipe) => InternalDataStucture::Void,
            AST::HaltStatement => InternalDataStucture::Void,
        }
    }

    fn tokenize_and_parse(
        &mut self,
        input: &str,
        previous_identifiers: Option<Vec<Identifier>>,
    ) -> Result<(Vec<AST>, Option<Vec<Identifier>>), ParseError> {
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
    fn set_previous_identifiers(&mut self, previous_identifiers: Option<Vec<Identifier>>) {
        self.previous_identifiers = previous_identifiers;
    }
    fn interpret(&mut self, input: &str) {
        let previous_identifiers = self.previous_identifiers.take();
        match self.tokenize_and_parse(input, previous_identifiers) {
            Ok((parsed_statements, new_identifiers)) => {
                for statement in parsed_statements {
                    match self.calculate_statement(&statement) {
                        InternalDataStucture::Void => continue,
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
